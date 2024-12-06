# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class RiskyBearRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_observations = 223
        num_actions = 8
    

    class terrain(LeggedRobotCfg.terrain):
        vertical_scale = 0.0005 # [m]


    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.2, 0.4] # min max [m/s]
            lin_vel_y = [-0.1, 0.1]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [0, 0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, .06] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "Upper0": +0.7854,   # [rad]
            "Upper1": -0.7854,   # [rad]
            "Upper2": +0.7854,   # [rad]
            "Upper3": -0.7854,   # [rad]
            "Lower0": +0.,     # [rad]
            "Lower1": +0.,   # [rad]
            "Lower2": +0.,     # [rad]
            "Lower3": +0.,   # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"Upper": 2.0, "Lower": 2.0}  # [N*m/rad]
        damping = {"Upper": .1, "Lower": .1}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/riskybear/urdf/riskybear.urdf"
        name = "riskybear"
        foot_name = "Lower"
        penalize_contacts_on = ["Upper0_Link", "Upper1_Link", "Upper2_Link", "Upper3_Link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            termination         = -0.1   # -0.1
            tracking_lin_vel    =  1.0   # 10.0
            tracking_ang_vel    =  0.5
            lin_vel_z           = -0.0
            ang_vel_xy          = -0.0
            orientation         = -0.01  # -0.3
            torques             = -1e-5  # -0.00001
            dof_vel             = -1e-4  # -0.00001
            dof_acc             = -0.0   # -2.0e-7
            base_height         = -0.0   # -5.0
            feet_air_time       =  0.5   # 1.0
            collision           = -0.001 # -1.
            feet_stumble        = -0.0
            action_rate         = -1e-4  # -0.01
            stand_still         = -0.0   # -0.00001

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma        = 0.05 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit    = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit    = 1.
        soft_torque_limit     = 1.
        base_height_target    = .1
        max_contact_force     = 100. # forces above this value are penalized

class RiskyBearRoughCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 128]
        critic_hidden_dims = [256, 128, 128]
        activation = "elu" # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "riskybear_rough"

  