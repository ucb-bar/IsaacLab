# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual policy wrapper for combining a frozen PPO base policy with SAC residual learning."""

import gymnasium as gym
import numpy as np
import os
import pickle
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper


class ResidualSb3VecEnvWrapper(Sb3VecEnvWrapper):
    """Wrapper for residual policy learning with a frozen PPO base policy.

    This wrapper extends Sb3VecEnvWrapper to:
    1. Load and freeze a pre-trained PPO policy
    2. Compute PPO actions for each observation
    3. Optionally augment observations with PPO actions
    4. Combine SAC's residual actions with PPO's base actions

    The final action applied to the environment is:
        final_action = clip(ppo_action + delta_scale * sac_action, action_low, action_high)
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        ppo_checkpoint_path: str,
        delta_scale: float = 0.3,
        include_ppo_action_in_obs: bool = True,
        ppo_vecnormalize_path: str | None = None,
        fast_variant: bool = True,
    ):
        """Initialize the residual wrapper.

        Args:
            env: The environment to wrap.
            ppo_checkpoint_path: Path to the pre-trained PPO model.
            delta_scale: Scale factor for SAC residual actions.
            include_ppo_action_in_obs: Whether to augment observations with PPO actions.
            ppo_vecnormalize_path: Optional path to PPO's VecNormalize pickle.
            fast_variant: Use fast variant for processing info.
        """
        # Initialize parent class first
        super().__init__(env, fast_variant=fast_variant)

        self.delta_scale = delta_scale
        self.include_ppo_action_in_obs = include_ppo_action_in_obs

        # Load frozen PPO policy
        print(f"[ResidualWrapper] Loading PPO policy from: {ppo_checkpoint_path}")
        self.ppo_model = PPO.load(ppo_checkpoint_path, device=self.sim_device)
        self.ppo_model.policy.set_training_mode(False)  # Set to eval mode

        # Load PPO's observation normalization if provided
        self.ppo_obs_normalizer = None
        if ppo_vecnormalize_path is not None and os.path.exists(ppo_vecnormalize_path):
            print(f"[ResidualWrapper] Loading PPO VecNormalize from: {ppo_vecnormalize_path}")
            with open(ppo_vecnormalize_path, "rb") as f:
                vecnorm_data = pickle.load(f)
            self.ppo_obs_rms = vecnorm_data.obs_rms
            self.ppo_clip_obs = vecnorm_data.clip_obs
            self.ppo_epsilon = vecnorm_data.epsilon
        else:
            self.ppo_obs_rms = None

        # Store action bounds for clipping
        self.action_low = self.action_space.low
        self.action_high = self.action_space.high

        # Cache for PPO actions (computed during reset and step)
        self._cached_ppo_actions = np.zeros((self.num_envs, self.action_space.shape[0]), dtype=np.float32)

        # Cache for SAC deltas (for logging)
        self._last_sac_delta = np.zeros((self.num_envs, self.action_space.shape[0]), dtype=np.float32)
        self._last_combined_action = np.zeros((self.num_envs, self.action_space.shape[0]), dtype=np.float32)

        # Modify observation space if we're including PPO actions
        if self.include_ppo_action_in_obs:
            original_obs_space = self.observation_space
            action_dim = self.action_space.shape[0]

            if isinstance(original_obs_space, gym.spaces.Box):
                # Extend observation space to include PPO action
                new_low = np.concatenate([original_obs_space.low, -np.ones(action_dim, dtype=np.float32)])
                new_high = np.concatenate([original_obs_space.high, np.ones(action_dim, dtype=np.float32)])
                self.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)
                print(f"[ResidualWrapper] Extended observation space: {original_obs_space.shape} -> {self.observation_space.shape}")
            elif isinstance(original_obs_space, gym.spaces.Dict):
                # For dict observations, add PPO action as a new key
                new_spaces = dict(original_obs_space.spaces)
                new_spaces["ppo_action"] = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
                )
                self.observation_space = gym.spaces.Dict(new_spaces)
                print("[ResidualWrapper] Added 'ppo_action' to dict observation space")

    def _normalize_obs_for_ppo(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using PPO's normalization stats if available."""
        if self.ppo_obs_rms is not None:
            obs = (obs - self.ppo_obs_rms.mean) / np.sqrt(self.ppo_obs_rms.var + self.ppo_epsilon)
            obs = np.clip(obs, -self.ppo_clip_obs, self.ppo_clip_obs)
        return obs

    def _compute_ppo_actions(self, obs: np.ndarray) -> np.ndarray:
        """Compute actions from the frozen PPO policy.

        Args:
            obs: The raw observations (before augmentation with PPO action).

        Returns:
            PPO actions as numpy array.
        """
        # Normalize observations for PPO if needed
        obs_for_ppo = self._normalize_obs_for_ppo(obs)

        # Convert to tensor for PPO
        obs_tensor = torch.as_tensor(obs_for_ppo, device=self.sim_device, dtype=torch.float32)

        with torch.no_grad():
            # Get deterministic action from PPO
            actions, _, _ = self.ppo_model.policy(obs_tensor, deterministic=True)
            ppo_actions = actions.cpu().numpy()

        return ppo_actions

    def _augment_obs_with_ppo_action(self, obs: np.ndarray | dict, ppo_actions: np.ndarray) -> np.ndarray | dict:
        """Augment observations with PPO actions.

        Args:
            obs: The original observations.
            ppo_actions: The PPO actions to append.

        Returns:
            Augmented observations.
        """
        if not self.include_ppo_action_in_obs:
            return obs

        if isinstance(obs, dict):
            obs = dict(obs)  # Copy to avoid modifying original
            obs["ppo_action"] = ppo_actions
        else:
            obs = np.concatenate([obs, ppo_actions], axis=-1)

        return obs

    def reset(self) -> np.ndarray | dict:
        """Reset the environment and compute initial PPO actions."""
        # Call parent reset - returns RAW observations
        obs = super().reset()

        # Compute PPO actions for the raw observations
        self._cached_ppo_actions = self._compute_ppo_actions(obs)

        # Augment observations with PPO actions for SAC
        augmented_obs = self._augment_obs_with_ppo_action(obs, self._cached_ppo_actions)

        return augmented_obs

    def step_async(self, actions: np.ndarray):
        """Receive SAC's residual actions and combine with PPO actions.

        Args:
            actions: SAC's residual actions (deltas).
        """
        # Convert to numpy if needed
        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)

        # Combine PPO base action with SAC residual
        # final_action = ppo_action + delta_scale * sac_delta
        combined_actions = self._cached_ppo_actions + self.delta_scale * actions

        # Clip to action bounds
        combined_actions = np.clip(combined_actions, self.action_low, self.action_high)

        # Store for logging/debugging
        self._last_sac_delta = actions
        self._last_combined_action = combined_actions

        # Call parent with combined actions
        super().step_async(combined_actions)

    def step_wait(self) -> VecEnvStepReturn:
        """Wait for step to complete and augment next observations with PPO actions."""
        # Get step results from parent
        # NOTE: Parent returns RAW observations (not augmented), so we use them directly
        obs, rewards, dones, infos = super().step_wait()

        # Compute PPO actions for the raw observations
        self._cached_ppo_actions = self._compute_ppo_actions(obs)

        # Augment observations with PPO actions for SAC
        augmented_obs = self._augment_obs_with_ppo_action(obs, self._cached_ppo_actions)

        # Fix terminal_observation in infos - must also be augmented for SB3 replay buffer
        for i, info in enumerate(infos):
            if info.get("terminal_observation") is not None:
                # The terminal observation from parent is raw (32 dims)
                # We need to augment it with PPO action to match SAC's observation space (39 dims)
                terminal_obs_raw = info["terminal_observation"]
                # Compute PPO action for this terminal observation
                terminal_ppo_action = self._compute_ppo_actions(terminal_obs_raw[np.newaxis, ...])[0]
                # Augment terminal observation
                if isinstance(terminal_obs_raw, dict):
                    info["terminal_observation"] = dict(terminal_obs_raw)
                    info["terminal_observation"]["ppo_action"] = terminal_ppo_action
                else:
                    info["terminal_observation"] = np.concatenate([terminal_obs_raw, terminal_ppo_action], axis=-1)

            # Add residual info for logging
            if info.get("episode") is not None:
                info["residual_magnitude"] = float(np.mean(np.abs(self._last_sac_delta[i])))

        return augmented_obs, rewards, dones, infos

