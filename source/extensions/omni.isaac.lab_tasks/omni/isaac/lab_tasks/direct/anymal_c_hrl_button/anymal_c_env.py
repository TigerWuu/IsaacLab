# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np
# import python queue
from collections import deque
import queue

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .anymal_c_env_cfg import AnymalCFlatEnvCfg, AnymalCRoughEnvCfg
# visualize target
from .targetVisualization import targetVisualization as targetVis
# visualize taining
import wandb

### add
from collections import deque
import pickle
import os
import cli_args
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
# from rsl_rl.runners import OnPolicyRunner
from .on_policy_runner import LoadPPOModel

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
### very ugly code..., cause I dont know how to use the decorator 
from omni.isaac.lab_tasks.direct.anymal_c_hrl.agents.rsl_rl_ppo_cfg import AnymalCFlatPPORunnerCfg, AnymalCRoughPPORunnerCfg

my_config = {
    "run_id": "button_1215_1000iter_HighAction&CenterMass_obs_0.2+0.3box_update",
    # "run_id": "test_3",
    "epoch_num": 1000,
    "description": "0 to 1000 epochs, command curriculum in x and y axis, change root frame position to (x,y,z), friction 1, average reward 13, clear buffer",
    "ex-max" : 0.7,
    "ex-step" : 0.1,
    "ex-threshold" : 13,
    "resample-time" : 6,
    # "xyz0": [[0.6, 0.8], [-0.2, -0.2], [0.0, 0.4]],
    # "xyz0": [[0.6, 0.8], [-0.2, 0.2], [0.0, 0.4]],
    "xyz0": [[0.6, 0.8], [-0.3, 0.1], [0.0, 0.4]], # button
    # "xyz0": [[0.6, 0.6], [-0.6, -0.6], [0.2, 0.2]], # for play.py
    # "xyz0": [[0.7, 0.7], [0.5, 0.5], [0.5, 0.5]], # test box
    # "xyz0": [[0.6, 0.8], [-0.2, 0.2], [0.0, 1.2]], # paper box
    "ex": 0.0,
    "touched": 0.08, # touched threshold
    # "foot" : "RF_FOOT", 
    # "foot" : "RF_FOOT", "LF_FOOT"
    "wandb" : True,
    "target_visual" : True,
}

class AnymalCEnv(DirectRLEnv):
    cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg

    def __init__(self, cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # define low level action space 
        self.low_level_actions = torch.zeros(self.num_envs, 12 , device=self.device)
        self.low_level_processed_actions = torch.zeros(self.num_envs, 12 , device=self.device)
        # Joint position command (deviation from default joint positions)
        # self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        ### add
        # discrete
        self.action_space = gym.spaces.Discrete(2) ### 2o4
        ###
        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # x,y,z target points in root frame
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)


        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "Re",
                "Rm",
                "Rn",
                "Rf",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._RF_FOOT_f, _ = self._contact_sensor.find_bodies("RF_FOOT")
        self._LF_FOOT_f, _ = self._contact_sensor.find_bodies("LF_FOOT")

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        self._undesired_contact_body_shank_ids, _ = self._contact_sensor.find_bodies(".*SHANK")

        # find the right front foot transform in world frame(e.g.)
        self._RF_FOOT, _ = self._robot.find_bodies("RF_FOOT")
        self._BASE, _ = self._robot.find_bodies("base")
        self._LF_FOOT, _ = self._robot.find_bodies("LF_FOOT")
        # init base frame origin (still confused about how to get this)
        # self.root_position = self._robot.data.default_root_state
        # self.root_position[:, :3] += self._terrain.env_origins
        # self.root_position = self._robot.data.body_pos_w[:, self._BASE[0], :3]
        self.root_position = self._terrain.env_origins[:] # wtf += will affect the default_root_state , array shallow copy?
        # for curriculum learning
        # (tiger) for curriculum learning
        self.x = my_config["xyz0"][0]
        self.y = my_config["xyz0"][1]
        self.z = my_config["xyz0"][2]
        self.ex = my_config["ex"]
        # set object
        self.rigid_poses = torch.zeros(self.num_envs, 7, device=self.device)

        # for marker visualization
        if my_config["target_visual"]:
            self.target = targetVis(scale=0.03, num_envs=self.num_envs)
        # for resampling target points
        self.resampled = torch.zeros(self.num_envs)

        # create a queue with buffer size 100
        self.buffer_size = 1000
        self.reward_buffer = deque(maxlen=self.buffer_size)

        ### add for hrl
        self.get_low_level_policy()
        self.high_level_actions = torch.zeros(self.num_envs, 1 , device=self.device)
        ###

        if my_config["wandb"]:
            run = wandb.init(
            project="RL_Final_hrl",
            config=my_config,
            id=my_config["run_id"]
        )
    ### add
    # @hydra_task_config("Isaac-Velocity-Flat-Anymal-C-Direct-hrl-v0", "rsl_rl_cfg_entry_point")
    def get_low_level_policy(self, agent_cfg = AnymalCFlatPPORunnerCfg()):
        ## need change 
        import argparse
        args_cli = argparse.Namespace()
        # args_cli.video = True
        # args_cli.video_length = 200
        # args_cli.video_interval = 2000
        args_cli.num_envs = 4096
        args_cli.seed = 42
        args_cli.max_iterations = 500
        args_cli.device = "cuda"

        # override configurations with non-hydra CLI arguments

        # agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )
        
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
    
        # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        ### multi-path
        # resume_paths = []
        # for i in range(4):
        #     low_policy = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        #     resume_paths.append(low_policy)
        ###
        resume_paths = ["/home/hyc/IsaacLab/saved_obs/model_.pt","/home/hyc/IsaacLab/saved_obs/model_.pt"]
        
        # create runner from rsl-rl
        # runner = OnPolicyRunner(agent_cfg.to_dict(), device=agent_cfg.device)
        runner_r = LoadPPOModel(agent_cfg.to_dict(), device=agent_cfg.device)
        runner_l = LoadPPOModel(agent_cfg.to_dict(), device=agent_cfg.device)
        
        ### multi-path
        self.low_level_policies = []
        
        # 2o4
        runner_r.load(resume_paths[0])
        runner_r.eval_mode()
        policy_r = runner_r.get_inference_policy()
        self.low_level_policies.append(policy_r)

        runner_l.load(resume_paths[1])
        runner_l.eval_mode()
        policy_l = runner_l.get_inference_policy()
        self.low_level_policies.append(policy_l)
        ###

    ###
    
    def _setup_scene(self):
        # add object
        self.object = RigidObject(self.cfg.cone_cfg)
        self.scene.rigid_objects["object"] = self.object
        
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        # print("observation : ", self.observations.shape)
        action_index_add = 0
        for i in range(self.num_envs):
            obs = self.observations["policy"][i]
            self.action_index = torch.argmax(actions[i]).item()
            self.high_level_actions[i] = self.action_index
            # print("action_index : ", action_index)
            low_level_action = self.low_level_policies[self.action_index](obs) 
            self.low_level_actions[i] = low_level_action
            # low_level_actions = torch.cat((low_level_actions, low_level_action), dim=0)

            # Log to W&B
            action_index_add += self.action_index

        # Log to W&B
        if my_config["wandb"]:
            wandb.log({
                "average_action_index": action_index_add/self.num_envs
            })

        # low_level_action = self.low_level_policies[action_index](self.observations) # how to setup discrete action (might be direct_rl_env?)
        # self._actions = self.low_level_actions.clone()

        self.low_level_processed_actions = self.cfg.action_scale * self.low_level_actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self.low_level_processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        # print("root_pos_world : ", self.root_position)
        # print("base_pos_world : ", self._robot.data.body_pos_w[:, self._BASE[0], :3])
        base_pos_root = self._robot.data.body_pos_w[:, self._BASE[0], :3] - self.root_position[:, :3]
        # print("base_pos_root : ", base_pos_root)
        
        # print("root_position : ", self._robot.data.root_pos_w[:, :3])
        # print("base_position : ", self._robot.data.body_pos_w[:, self._BASE[0], :3])
        # print("basePos - rootPos : ", self._robot.data.body_pos_w[:, self._BASE[0], :3] - self._robot.data.root_pos_w[:, :3])
        # print("command_ : ", self._commands)

        #### in base frame ####
        self._commands_base = self._commands - base_pos_root # command : root to base
        
        ### not sure
        contacts_forces = self._contact_sensor.data.net_forces_w
        self._rf_norm = torch.norm(contacts_forces[:, self._RF_FOOT_f[0]], dim=1).unsqueeze(-1)
        contacts_forces = self._contact_sensor.data.net_forces_w
        self._lf_norm = torch.norm(contacts_forces[:, self._LF_FOOT_f[0]], dim=1).unsqueeze(-1)

        # print("RF force norm : ", torch.norm(contacts_forces[:, self._RF_FOOT_f[0]], dim=1).unsqueeze(-1)) 

        height_data = None
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands_base, # base frame 
                    # self._commands, # root frame
                    self._robot.data.joint_pos,
                    self._robot.data.joint_vel,
                    # height_data,
                    # self._actions, ### add (delete)
                    self.low_level_actions,
                    self._rf_norm,
                    self._lf_norm,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        ### add for high level policy
        # obs_RF_FOOT, _ = self._robot.find_bodies("RF_FOOT")
        # obs_LF_FOOT, _ = self._robot.find_bodies("LF_FOOT")
        RF_FOOT_pos_base = self._robot.data.body_pos_w[:, self._RF_FOOT[0], :3] - self._robot.data.body_pos_w[:, self._BASE[0], :3]
        LF_FOOT_pos_base = self._robot.data.body_pos_w[:, self._LF_FOOT[0], :3] - self._robot.data.body_pos_w[:, self._BASE[0], :3]
        mass_center_base = self._robot.data.body_pos_w[:, self._BASE[0], :3]
        mass_center_deviation = torch.norm((mass_center_base-self.root_position), dim=1)

        ### mass_center_deviation [4096] --> [4096,1]
        mass_center_deviation = mass_center_deviation.unsqueeze(1)
        # print("mass_center_deviation shape:", mass_center_deviation.shape)
        # print("LF_FOOT_pos_base shape:", LF_FOOT_pos_base.shape)
        # print("self.high_level_actions shape:", self.high_level_actions.shape)
        # print("self._commands_base shape:", self._commands_base.shape)
        
        obs_h = torch.cat(
            [
                tensor
                for tensor in (

                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands_base, # base frame 
                    # self._commands, # root frame
                    self._robot.data.joint_pos,
                    self._robot.data.joint_vel,
                    # height_data,
                    # self._actions, 
                    # self.low_level_actions,
                    self.high_level_actions,

                    ### add for high level policy
                    mass_center_deviation,
                    LF_FOOT_pos_base,
                    RF_FOOT_pos_base,

                    ### not sure
                    self._rf_norm,
                    self._lf_norm,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        self.observations = {"policy": obs}
        obs_h_return  = {"policy": obs_h}
        ###
        return obs_h_return

    def _get_rewards(self) -> torch.Tensor:
        # resample the target point every half episode
        resampled_ids_cand = torch.where(self.episode_length_buf >= self.max_episode_length/2, 1.0, 0).nonzero() #.squeeze()
        # print("resampled_ids : ", resampled_ids)
        resampled_ids = []
        for i in resampled_ids_cand:
            if self.resampled[i.item()] == 0:
                self.resampled[i.item()] = 1
                resampled_ids.append(i.item())
        
        if len(resampled_ids) > 0:
            resampled_ids = torch.tensor(resampled_ids, device=self.device)
            x = np.random.uniform(self.x[0], self.x[1]+2*self.ex)
            y = np.random.uniform(self.y[0]-self.ex, self.y[1]+self.ex)
            # z = np.random.uniform(self.z[0], self.z[1])
            if self.z[1]+2*self.ex<1.1:
                z = np.random.uniform(self.z[0], self.z[1]+2*self.ex)
            else:
                z = np.random.uniform(self.z[0], 1.2)
            self._commands[resampled_ids] = torch.tensor([x, y, z], device=self.device)

            ### button
            root_state = self.object.data.default_root_state.clone()[resampled_ids]
            root_state[:, :3] = torch.tensor([x, y, z], device=self.device)+self.scene.env_origins[resampled_ids]
            self.object.write_root_pose_to_sim(root_state[:, :7], resampled_ids)
        
        
        ### Re ###
        # foot position(w1)
        # world frame -> base frame
        

        #### in base frame ####
        # LF_FOOT or RF_FOOT
        RF_FOOT_pos_base = self._robot.data.body_pos_w[:, self._RF_FOOT[0], :3] - self._robot.data.body_pos_w[:, self._BASE[0], :3]
        RF_foot_pos_deviation = torch.norm((RF_FOOT_pos_base-self._commands_base[:, :3]), dim=1)
      
        LF_FOOT_pos_base = self._robot.data.body_pos_w[:, self._LF_FOOT[0], :3] - self._robot.data.body_pos_w[:, self._BASE[0], :3]
        LF_foot_pos_deviation = torch.norm((LF_FOOT_pos_base-self._commands_base[:, :3]), dim=1)

        HL_actions = self.high_level_actions.squeeze()
        foot_pos_deviation = torch.where(HL_actions == 0, RF_foot_pos_deviation, LF_foot_pos_deviation)

        mass_center_base = self._robot.data.body_pos_w[:, self._BASE[0], :3]
        mass_center_deviation = torch.norm((mass_center_base-self.root_position), dim=1)

        #### in root frame ####
        
        # RF_FOOT_pos_root = self._robot.data.body_pos_w[:, self._RF_FOOT[0], :3] - self.root_position[:, :3]
        # RF_foot_pos_deviation = torch.norm((RF_FOOT_pos_root-self._commands[:, :3]), dim=1)
        # LF_FOOT_pos_root = self._robot.data.body_pos_w[:, self._LF_FOOT[0], :3] - self.root_position[:, :3]
        # LF_foot_pos_deviation = torch.norm((LF_FOOT_pos_root-self._commands[:, :3]), dim=1)
        ### mass deviation (dont know the function)
        # mass_deviation = torch.norm(self._robot.data.mass_center_w[:, :3] - self._commands[:, :3], dim=1)
        
        # target visualization
        if my_config["target_visual"]:
            self.target.set_marker_position(self._commands, self.root_position)
            self.target.check_marker_touched(foot_pos_deviation, my_config["touched"])
            # if self.action_index == 0:
            #     self.target.check_marker_touched(RF_foot_pos_deviation, my_config["touched"])
            # elif self.action_index == 1:
            #     self.target.check_marker_touched(LF_foot_pos_deviation, my_config["touched"])
            self.target.visualize()
        ### Rn ###
        # joint velocity(w2)
        joint_vel = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        # joint torques(w4)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration(w3)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate(w5)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # nc - number of collisions(w6)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        contacts_forces = self._contact_sensor.data.net_forces_w
        # print("RF force : ", contacts_forces[:, self._RF_FOOT_f[0]]) 
        # print("LF force : ", contacts_forces[:, self._LF_FOOT_f[0]]) 

        ### button
        rf_force = contacts_forces[:, self._RF_FOOT_f[0]]
        rf_force_magnitude = torch.norm(rf_force, dim=1)
        print("RF force magnitude : ", torch.norm(rf_force, dim=1)) 
        target_force = 10.0  #target force
        rf_force_deviation = torch.abs(target_force - rf_force_magnitude) # force deviation

        lf_force = contacts_forces[:, self._LF_FOOT_f[0]]
        lf_force_magnitude = torch.norm(lf_force, dim=1)
        print("LF force magnitude : ", torch.norm(lf_force, dim=1)) 
        target_force = 10.0  #target force
        lf_force_deviation = torch.abs(target_force - lf_force_magnitude) # force deviation

        force_deviation = torch.where(HL_actions == 0, rf_force_deviation, lf_force_deviation)


        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        is_contact_shank = (    
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_shank_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts_shank = torch.sum(is_contact_shank, dim=1)
        contacts_thigh = torch.sum(is_contact, dim=1)
        contacts = contacts_shank + contacts_thigh
        # print("contacts", contacts_shank, contacts_thigh)

        # termination penalty(w7)
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._BASE], dim=-1), dim=1)[0] > 1.0, dim=1)
        
        rewards = {
                "Re": self.cfg.w1* torch.exp(-(foot_pos_deviation/self.cfg.sigma))* self.step_dt,
                "Rm": self.cfg.w1* torch.exp(-(mass_center_deviation/self.cfg.sigma))* self.step_dt,
                "Rn": (self.cfg.w2 * joint_vel + self.cfg.w3 * joint_accel + self.cfg.w4 * joint_torques + self.cfg.w5 * action_rate + self.cfg.w6 * contacts + self.cfg.w7 * died)* self.step_dt,
                "Rf": self.cfg.w8 * torch.exp(-(force_deviation/self.cfg.sigma_f)) * self.step_dt, 
            }
        
        # if self.action_index == 0:
        #     # tensor_zero = torch.zeros(4096)
        #     # tensor_gpu = tensor_zero.to('cuda:0')
        #     # tensor_negative_ten = torch.full((4096,), -10)
        #     # tensor_gpu_1 = tensor_negative_ten.to('cuda:0')
        #     rewards = {
        #         # "Re" : tensor_gpu,
        #         # "Rm" : tensor_gpu,
        #         # "Rn" : tensor_gpu_1,
        #         "Re": self.cfg.w1* torch.exp(-(RF_foot_pos_deviation/self.cfg.sigma))* self.step_dt,
        #         "Rm": self.cfg.w1* torch.exp(-(mass_center_deviation/self.cfg.sigma))* self.step_dt,
        #         "Rn": (self.cfg.w2 * joint_vel + self.cfg.w3 * joint_accel + self.cfg.w4 * joint_torques + self.cfg.w5 * action_rate + self.cfg.w6 * contacts + self.cfg.w7 * died)* self.step_dt, 
        #     }
        # elif self.action_index == 1:
        #     rewards = {
        #         "Re": self.cfg.w1* torch.exp(-(LF_foot_pos_deviation/self.cfg.sigma))* self.step_dt,
        #         "Rm": self.cfg.w1* torch.exp(-(mass_center_deviation/self.cfg.sigma))* self.step_dt,
        #         "Rn": (self.cfg.w2 * joint_vel + self.cfg.w3 * joint_accel + self.cfg.w4 * joint_torques + self.cfg.w5 * action_rate + self.cfg.w6 * contacts + self.cfg.w7 * died)* self.step_dt,
        #     }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        ### check center of mass

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        for i in env_ids:
            self.resampled[i.item()] = 0

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self.object.reset(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        ### add
        # self._actions[env_ids] = 0.0
        self._actions[env_ids] = 0
        # self._previous_actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0
        ###
        # Sample new commands
        # first curriculum
        x = np.random.uniform(self.x[0], self.x[1]+2*self.ex)
        y = np.random.uniform(self.y[0]-self.ex, self.y[1]+self.ex)
        # z = np.random.uniform(self.z[0], self.z[1])
        if self.z[1]+2*self.ex < 1.1:
            z = np.random.uniform(self.z[0], self.z[1]+2*self.ex)
        else:
            z = np.random.uniform(self.z[0], 1.2)

        ### button
        self._commands[env_ids] = torch.tensor([x, y, z], device=self.device)
        # reset object position
        root_state = self.object.data.default_root_state.clone()[env_ids]
        # add xyz to root state[env_ids]
        root_state[:, :3] = torch.tensor([x, y, z], device=self.device)+self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(root_state[:, :7], env_ids)

        # target visualization
        if my_config["target_visual"]:
            self.target.set_marker_position(self._commands, self.root_position)
            self.target.reset_indices(env_ids)
            self.target.visualize()

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        # curriculm learning
        # put the reward into the queue
        self.reward_buffer.append(extras["Episode_Reward/Re"])
        # if the queue is full
        if len(self.reward_buffer) == self.buffer_size:
            # get the average reward
            avg_reward = sum(self.reward_buffer)/self.buffer_size
            # if the average reward is larger than the threshold, increase the curriculum
            if avg_reward.item() > my_config["ex-threshold"]:   # distance < 0.08
                if self.ex < my_config["ex-max"]:
                    self.ex += my_config["ex-step"]
                    self.reward_buffer.clear()  
        else:
            avg_reward = torch.tensor(0.0, device=self.device)
        
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
        
        if my_config["wandb"]:
            # wandb logging
            wandb.log(self.extras["log"]) 
            wandb.log({"avg_reward_100": avg_reward.item()})
            wandb.log({"curriculum": self.ex})
