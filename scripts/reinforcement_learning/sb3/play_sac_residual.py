# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained residual SAC agent on top of a frozen PPO policy.

This loads both the PPO base policy and the SAC residual policy, combining them
to produce the final actions: final_action = ppo_action + delta_scale * sac_delta
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained residual SAC agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
# Residual-specific arguments
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to the SAC residual model checkpoint (.zip file).",
)
parser.add_argument(
    "--ppo_checkpoint",
    type=str,
    default=None,
    help="Path to the pre-trained PPO model checkpoint (.zip file). If not provided, will try to load from residual.yaml.",
)
parser.add_argument(
    "--ppo_vecnormalize",
    type=str,
    default=None,
    help="Path to the PPO VecNormalize pickle file (optional).",
)
parser.add_argument(
    "--delta_scale",
    type=float,
    default=None,
    help="Scale factor for SAC residual actions. If not provided, will try to load from residual.yaml.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch
import yaml

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.sb3 import process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

# Import the shared residual wrapper
from residual_wrapper import ResidualSb3VecEnvWrapper


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with residual SAC agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # set the environment seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3_residual", train_task_name)
    log_root_path = os.path.abspath(log_root_path)

    # Find SAC checkpoint
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint, sort_alpha=False)
    else:
        checkpoint_path = args_cli.checkpoint

    log_dir = os.path.dirname(checkpoint_path)
    print(f"[INFO] SAC checkpoint: {checkpoint_path}")
    print(f"[INFO] Log directory: {log_dir}")

    # Try to load residual config from training
    residual_cfg_path = os.path.join(log_dir, "params", "residual.yaml")
    ppo_checkpoint = args_cli.ppo_checkpoint
    ppo_vecnormalize = args_cli.ppo_vecnormalize
    delta_scale = args_cli.delta_scale

    if os.path.exists(residual_cfg_path):
        print(f"[INFO] Loading residual config from: {residual_cfg_path}")
        with open(residual_cfg_path, "r") as f:
            residual_cfg = yaml.safe_load(f)
        # Use saved values if not provided via CLI
        if ppo_checkpoint is None:
            ppo_checkpoint = residual_cfg.get("ppo_checkpoint")
        if ppo_vecnormalize is None:
            ppo_vecnormalize = residual_cfg.get("ppo_vecnormalize")
        if delta_scale is None:
            delta_scale = residual_cfg.get("delta_scale", 0.3)
        include_ppo_action_in_obs = residual_cfg.get("include_ppo_action_in_obs", True)
    else:
        print("[WARNING] No residual.yaml found, using CLI arguments only")
        include_ppo_action_in_obs = True
        if delta_scale is None:
            delta_scale = 0.3

    # Validate PPO checkpoint
    if ppo_checkpoint is None:
        raise ValueError("PPO checkpoint path must be provided via --ppo_checkpoint or in residual.yaml")
    if not os.path.exists(ppo_checkpoint):
        raise FileNotFoundError(f"PPO checkpoint not found: {ppo_checkpoint}")

    print(f"[INFO] PPO checkpoint: {ppo_checkpoint}")
    print(f"[INFO] Delta scale: {delta_scale}")

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap with residual SB3 wrapper
    print(f"[INFO] Creating residual environment with delta_scale={delta_scale}")
    env = ResidualSb3VecEnvWrapper(
        env,
        ppo_checkpoint_path=ppo_checkpoint,
        delta_scale=delta_scale,
        include_ppo_action_in_obs=include_ppo_action_in_obs,
        ppo_vecnormalize_path=ppo_vecnormalize,
        fast_variant=not args_cli.keep_all_info,
    )

    # Load SAC's VecNormalize if it exists
    vec_norm_path = checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
    vec_norm_path = Path(vec_norm_path)

    if vec_norm_path.exists():
        print(f"[INFO] Loading SAC normalization: {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        # Do not update at test time
        env.training = False
        env.norm_reward = False
    elif "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=False,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        )

    # Load SAC agent
    print(f"[INFO] Loading SAC checkpoint from: {checkpoint_path}")
    agent = SAC.load(checkpoint_path, env, print_system_info=True)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    timestep = 0

    print("[INFO] Starting residual policy playback...")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping - SAC outputs residual actions
            actions, _ = agent.predict(obs, deterministic=True)
            # env stepping - wrapper combines with PPO action
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
