# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AnymalCFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    ### add
    # resume = True
    # load_run = "2024-12-13_17-43-30"  # The run directory to load. Default is ".*" (all).
    # load_checkpoint = "model_499.pt" # The checkpoint file to load. Default is ``"model_.*.pt"`` (all).
    ###
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 100
    experiment_name = "anymal_c_flat_direct_hrl"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        ### add
        # clip_param=0.5,
        # entropy_coef=0.1,
        ###
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnymalCRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    ### add
    # load_run = ".*" # The run directory to load. Default is ".*" (all).
    # load_checkpoint = "model_.*.pt" # The checkpoint file to load. Default is ``"model_.*.pt"`` (all).
    ###
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "anymal_c_rough_direct_hrl"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )