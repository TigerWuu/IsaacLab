from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np
# import python queue
from collections import deque
import queue
import time
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .anymal_c_env_cfg import AnymalCFlatEnvCfg, AnymalCRoughEnvCfg
# visualize target
from .targetVisualization import targetVisualization as targetVis
# visualize taining
import wandb

import csv
import os

# Create a directory for saving results
os.makedirs("results", exist_ok=True)
execution_time_file = "./results/results_rl.csv"

# Write headers to the file if it doesn't exist
if not os.path.exists(execution_time_file):
    with open(execution_time_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Exacution_Time", "Avg_Time"])


my_config = {
    "run_id": "Quadruped_tripod_curri-01-xyz_resampled-6s_AvgRe-13_frame-base_friction-1-paper-box_buffer-1000",
    "epoch_num": 12000,
    "description": "0 to 12000 epochs, command curriculum in x and y axis, change root frame position to (x,y,z), friction 1, average reward 13, clear buffer",
    "ex-max" : 0.7,
    "ex-step" : 0.1,
    "ex-threshold" : 13,
    "resample-time" : 6,
    # "xyz0": [[0.6, 0.8], [-0.2, 0.2], [0.0, 0.4]],
    "xyz0": [[0.6, 0.8], [-0.2, 0.2], [0.0, 1.2]],
    "ex": 0.4,
}

class AnymalCEnv(DirectRLEnv):
    cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg

    def __init__(self, cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
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
                "Rn",
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
        self._LF_FOOT, _ = self._robot.find_bodies("LF_FOOT")
        self._BASE, _ = self._robot.find_bodies("base")
        # init base frame origin (still confused about how to get this)
        # self.root_position = self._robot.data.default_root_state[:, :3]
        self.root_position = self._terrain.env_origins[:] # wtf += will affect the default_root_state , array shallow copy?
        # self.root_position = self._robot.data.body_pos_w[:, self._BASE[0], :3]
        # print("root_position : ", self.root_position)

        # (tiger) for curriculum learning
        self.x = my_config["xyz0"][0]
        self.y = my_config["xyz0"][1]
        self.z = my_config["xyz0"][2]
        self.ex = my_config["ex"]
        
        # (chang)for curriculum learning
        # self.x = [0.6, 0.8]
        # self.y = [-0.2, 0.2]
        # self.z = [0.4, 1.1]
        # self.ex = 0.0


        # for marker visualization
        self.target = targetVis(scale=0.03, num_envs=self.num_envs)

        # for resampling target points
        self.resampled = torch.zeros(self.num_envs)
        
        # create a queue with buffer size 100
        self.buffer_size = 1000
        self.reward_buffer = deque(maxlen=self.buffer_size)

        # time
        self.st = 0
        self.et = 0
        self.frist_touch_check = True
        self.avg_time = []
        self.avg10times = 0
        self.boxex = 0
        # # wandb logging
        # run = wandb.init(
        #     project="RL_Final",
        #     config=my_config,
        #     id=my_config["run_id"]
        # )
    
    def _setup_scene(self):
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
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        # print("root_pos_world : ", self.root_position)
        # print("base_pos_world : ", self._robot.data.body_pos_w[:, self._BASE[0], :3])
        # print("base_pos_root : ", base_pos_root)
        
        # print("root_position : ", self._robot.data.root_pos_w[:, :3])
        # print("base_position : ", self._robot.data.body_pos_w[:, self._BASE[0], :3])
        # print("basePos - rootPos : ", self._robot.data.body_pos_w[:, self._BASE[0], :3] - self._robot.data.root_pos_w[:, :3])
        # print("command_ : ", self._commands)

        #### in base frame ####
        base_pos_root = self._robot.data.body_pos_w[:, self._BASE[0], :3] - self.root_position[:, :3]
        self._commands_base = self._commands - base_pos_root # command : root to base
        
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
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # # resample the target point every half episode
        # resampled_ids_cand = torch.where(self.episode_length_buf >= self.max_episode_length/2, 1.0, 0).nonzero()
        # # print("resampled_ids : ", resampled_ids)
        # resampled_ids = []
        
        # for i in resampled_ids_cand:
        #     if self.resampled[i.item()] == 0:
        #         self.resampled[i.item()] = 1
        #         resampled_ids.append(i.item())
        
        # if len(resampled_ids) > 0:
        #     resampled_ids = torch.tensor(resampled_ids, device=self.device)
        #     x = np.random.uniform(self.x[0], self.x[1]+2*self.ex)
        #     y = np.random.uniform(self.y[0]-self.ex, self.y[1]+self.ex)  #, self.y[1]+self.ex
        #     z = np.random.uniform(self.z[0], self.z[1])
        #     # x = 0.6
        #     # y = 0.5
        #     # z = 0.4
        #     # if self.z[1]<1.1:
        #     #     z = np.random.uniform(self.z[0], self.z[1]+2*self.ex)
        #     # else:
        #     #     z = np.random.uniform(self.z[0], self.z[1])
        #     self._commands[resampled_ids] = torch.tensor([x, y, z], device=self.device)
        
        
        ### Re ###
        # foot position(w1)
        # world frame -> base frame
        

        #### in base frame ####
        RF_FOOT_pos_base = self._robot.data.body_pos_w[:, self._RF_FOOT[0], :3] - self._robot.data.body_pos_w[:, self._BASE[0], :3]
        foot_pos_deviation = torch.norm((RF_FOOT_pos_base-self._commands_base[:, :3]), dim=1)
      
        #### in root frame ####
        # RF_FOOT_pos_root = self._robot.data.body_pos_w[:, self._RF_FOOT[0], :3] - self.root_position[:, :3]
        # foot_pos_deviation = torch.norm((RF_FOOT_pos_root-self._commands[:, :3]), dim=1)

        # target visualization
        self.target.set_marker_position(self._commands, self.root_position)
        self.target.check_marker_touched(foot_pos_deviation, 0.08)
        if foot_pos_deviation<0.08:
            if self.frist_touch_check:
                self.et = time.perf_counter()
                exacution_time = self.et-self.st
                print(f'touch time {exacution_time}##############################################################################')
                self.avg_time.append(exacution_time)
                self.avg10times += 1
                self.frist_touch_check = False

                # Save to CSV
                with open(execution_time_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    avg_time_to_save = np.mean(self.avg_time) if self.avg_time else 0
                    writer.writerow([exacution_time, avg_time_to_save])
        # self.target.visualize()
        
        # print("foot_pos_deviation : ", foot_pos_deviation)
        # print("RF_FOOT_pos_root : ", RF_FOOT_pos_root)
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
            "Rn": (self.cfg.w2 * joint_vel + self.cfg.w3 * joint_accel + self.cfg.w4 * joint_torques + self.cfg.w5 * action_rate + self.cfg.w6 * contacts + self.cfg.w7 * died)* self.step_dt, 
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # for resampling target points
        # for i in env_ids:
        #     self.resampled[i.item()] = 0

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        # first curriculum
        x = np.random.uniform(self.x[0]+0.2*self.boxex, self.x[1]+0.2*self.boxex)
        y = np.random.uniform(0, self.y[1]+0.1*self.boxex) # , self.y[1]+self.ex
        # y = np.random.uniform(self.y[0]-0.1*self.boxex, 0) # , self.y[1]+self.ex
        z = np.random.uniform(self.z[0], self.z[1])
        # x = 0.6
        # y = -0.5
        # z = 0.4
        # if self.z[1]<1.1:
        #     z = np.random.uniform(self.z[0], self.z[1]+2*self.ex)
        # else:
        #     z = np.random.uniform(self.z[0], self.z[1])

        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(0.3, 0.6)
        self._commands[env_ids] = torch.tensor([x, y, z], device=self.device)
        # target visualization
        self.target.set_marker_position(self._commands, self.root_position)
        self.target.reset_indices(env_ids)
        self.target.visualize()

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        # print("default_root_state : ", default_root_state[:, :3])
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Reset base frame origin
        # self.root_position = self._robot.data.body_pos_w[:, self._BASE[0], :3] # wrong for initialize every time

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
        # # wandb logging
        # wandb.log(self.extras["log"]) 
        # wandb.log({"avg_reward_100": avg_reward.item()})
        # wandb.log({"curriculum": self.ex})

        self.st = time.perf_counter()
        self.frist_touch_check = True
        if self.avg10times == 10:
            avg_time = np.mean(self.avg_time)
            print(f'Avg time {self.boxex} : {avg_time}')

            # Save the average time to CSV
            with open(execution_time_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"Batch Avg{self.boxex}", avg_time])

            self.avg10times = 0 #reset
            self.avg_time = []
            self.boxex += 1
