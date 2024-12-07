### need check training setting
from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
# from rsl_rl.utils import store_code_state
# from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlPpoActorCriticCfg,
#     RslRlPpoAlgorithmCfg,
# )
from .rsl_rl_config_mine import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


class OnPolicyRunner_no_env:
    def __init__(self, log_dir=None, device="cpu"):
        ### move setting from rsl_rl_ppo_cfg.py
        # train_cfg = RslRlOnPolicyRunnerCfg()
        train_cfg = {}
        self.cfg = train_cfg

        self.cfg["num_actions"] = 12
        self.cfg["num_steps_per_env"] = 24
        # max_iterations = 500
        self.cfg["save_interval"] = 100
        # experiment_name = "anymal_c_flat_direct_tripod_hrl"
        self.cfg["empirical_normalization"] = False
        train_cfg["policy"] = {
            'init_noise_std': 1.0, 
            'actor_hidden_dims': [128, 128, 128], 
            'critic_hidden_dims': [128, 128, 128], 
            'activation': 'elu'
        }
        
        # RslRlPpoActorCriticCfg(
        #     init_noise_std=1.0,
        #     actor_hidden_dims=[128, 128, 128],
        #     critic_hidden_dims=[128, 128, 128],
        #     activation="elu",
        # )

        train_cfg["algorithm"] = {
            'value_loss_coef': 1.0, 
            'use_clipped_value_loss': True, 
            'clip_param': 0.2, 
            'entropy_coef': 0.01, 
            'num_learning_epochs': 5, 
            'num_mini_batches': 4, 
            'learning_rate': 0.001, 
            'schedule': 'adaptive', 
            'gamma': 0.99, 
            'lam': 0.95, 
            'desired_kl': 0.01, 
            'max_grad_norm': 1.0
            }
        
        # RslRlPpoAlgorithmCfg(
        #     value_loss_coef=1.0,
        #     use_clipped_value_loss=True,
        #     clip_param=0.2,
        #     entropy_coef=0.01,
        #     num_learning_epochs=5,
        #     num_mini_batches=4,
        #     learning_rate=1.0e-3,
        #     schedule="adaptive",
        #     gamma=0.99,
        #     lam=0.95,
        #     desired_kl=0.01,
        #     max_grad_norm=1.0,
        # )
        
        ###
        
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        # self.env 被移除，因為你不再使用環境
        # obs, extras = self.env.get_observations()
        num_obs = 48 # obs.shape[1]
        num_critic_obs = num_obs  # 假設沒有特別的 critic 觀察

        actor_critic_class = ActorCritic
        # actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        print("here---------",self.policy_cfg)
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            num_obs, num_critic_obs, self.cfg["num_actions"], **self.policy_cfg
        ).to(self.device)
        
        alg_class = PPO
        # alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        
        # 自定義參數
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        
        # init storage and model
        self.alg.init_storage(
            num_envs=64, ### need check training setting 
            # num_steps_per_env=self.num_steps_per_env,
            # obs_shape=[num_obs],
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[num_obs],
            critic_obs_shape=[num_critic_obs],
            action_shape=[self.cfg["num_actions"]],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=True)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    
    def get_inference_policy(self, device=None):
        # self.eval_mode()  # switch to evaluation mode (dropout for example)
        
        if device is not None:
            self.alg.actor_critic.to(device)

        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy


