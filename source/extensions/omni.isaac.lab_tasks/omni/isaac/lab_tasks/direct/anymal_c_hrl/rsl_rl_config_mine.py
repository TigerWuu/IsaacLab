from typing import List
from typing import Optional
from typing import Literal

class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = None
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: List[int] = None
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: List[int] = None
    """The hidden dimensions of the critic network."""

    activation: str = None
    """The activation function for the actor and critic networks."""

    def __init__(self, class_name="ActorCritic", init_noise_std=None, actor_hidden_dims=None, 
                 critic_hidden_dims=None, activation=None):
        self.class_name = class_name
        self.init_noise_std = init_noise_std
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.activation = activation

    @classmethod
    def from_dict(cls, data: dict):
        """Create an instance from a dictionary."""
        return cls(
            class_name=data.get("class_name", "ActorCritic"),
            init_noise_std=data.get("init_noise_std"),
            actor_hidden_dims=data.get("actor_hidden_dims"),
            critic_hidden_dims=data.get("critic_hidden_dims"),
            activation=data.get("activation")
        )

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {
            "class_name": self.class_name,
            "init_noise_std": self.init_noise_std,
            "actor_hidden_dims": self.actor_hidden_dims,
            "critic_hidden_dims": self.critic_hidden_dims,
            "activation": self.activation
        }

    def copy(self):
        """Return a new instance with the same attributes."""
        return RslRlPpoActorCriticCfg(
            class_name=self.class_name,
            init_noise_std=self.init_noise_std,
            actor_hidden_dims=self.actor_hidden_dims,
            critic_hidden_dims=self.critic_hidden_dims,
            activation=self.activation
        )
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: Optional[float] = None
    """The coefficient for the value loss."""

    use_clipped_value_loss: Optional[bool] = None
    """Whether to use clipped value loss."""

    clip_param: Optional[float] = None
    """The clipping parameter for the policy."""

    entropy_coef: Optional[float] = None
    """The coefficient for the entropy loss."""

    num_learning_epochs: Optional[int] = None
    """The number of learning epochs per update."""

    num_mini_batches: Optional[int] = None
    """The number of mini-batches per update."""

    learning_rate: Optional[float] = None
    """The learning rate for the policy."""

    schedule: Optional[str] = None
    """The learning rate schedule."""

    gamma: Optional[float] = None
    """The discount factor."""

    lam: Optional[float] = None
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: Optional[float] = None
    """The desired KL divergence."""

    max_grad_norm: Optional[float] = None
    """The maximum gradient norm."""

    def __init__(self, class_name="PPO", value_loss_coef=None, use_clipped_value_loss=None,
                 clip_param=None, entropy_coef=None, num_learning_epochs=None, num_mini_batches=None,
                 learning_rate=None, schedule=None, gamma=None, lam=None, desired_kl=None, max_grad_norm=None):
        self.class_name = class_name
        self.value_loss_coef = value_loss_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.gamma = gamma
        self.lam = lam
        self.desired_kl = desired_kl
        self.max_grad_norm = max_grad_norm

    @classmethod
    def from_dict(cls, data: dict):
        """Create an instance from a dictionary."""
        return cls(
            class_name=data.get("class_name", "PPO"),
            value_loss_coef=data.get("value_loss_coef"),
            use_clipped_value_loss=data.get("use_clipped_value_loss"),
            clip_param=data.get("clip_param"),
            entropy_coef=data.get("entropy_coef"),
            num_learning_epochs=data.get("num_learning_epochs"),
            num_mini_batches=data.get("num_mini_batches"),
            learning_rate=data.get("learning_rate"),
            schedule=data.get("schedule"),
            gamma=data.get("gamma"),
            lam=data.get("lam"),
            desired_kl=data.get("desired_kl"),
            max_grad_norm=data.get("max_grad_norm")
        )

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {
            "class_name": self.class_name,
            "value_loss_coef": self.value_loss_coef,
            "use_clipped_value_loss": self.use_clipped_value_loss,
            "clip_param": self.clip_param,
            "entropy_coef": self.entropy_coef,
            "num_learning_epochs": self.num_learning_epochs,
            "num_mini_batches": self.num_mini_batches,
            "learning_rate": self.learning_rate,
            "schedule": self.schedule,
            "gamma": self.gamma,
            "lam": self.lam,
            "desired_kl": self.desired_kl,
            "max_grad_norm": self.max_grad_norm
        }

    def copy(self):
        """Return a new instance with the same attributes."""
        return RslRlPpoAlgorithmCfg(
            class_name=self.class_name,
            value_loss_coef=self.value_loss_coef,
            use_clipped_value_loss=self.use_clipped_value_loss,
            clip_param=self.clip_param,
            entropy_coef=self.entropy_coef,
            num_learning_epochs=self.num_learning_epochs,
            num_mini_batches=self.num_mini_batches,
            learning_rate=self.learning_rate,
            schedule=self.schedule,
            gamma=self.gamma,
            lam=self.lam,
            desired_kl=self.desired_kl,
            max_grad_norm=self.max_grad_norm
        )
    
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: Optional[int] = None
    """The number of steps per environment per update."""

    max_iterations: Optional[int] = None
    """The maximum number of iterations."""

    empirical_normalization: Optional[bool] = None
    """Whether to use empirical normalization."""

    policy: Optional[RslRlPpoActorCriticCfg] = None
    """The policy configuration."""

    algorithm: Optional[RslRlPpoAlgorithmCfg] = None
    """The algorithm configuration."""

    save_interval: Optional[int] = None
    """The number of iterations between saves."""

    experiment_name: Optional[str] = None
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string."""

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab."""

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all)."""

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is "model_.*.pt"."""

    def __init__(self, seed=42, device="cuda:0", num_steps_per_env=None, max_iterations=None,
                 empirical_normalization=None, policy=None, algorithm=None, save_interval=None, 
                 experiment_name=None, run_name="", logger="tensorboard", neptune_project="isaaclab", 
                 wandb_project="isaaclab", resume=False, load_run=".*", load_checkpoint="model_.*.pt"):
        self.seed = seed
        self.device = device
        self.num_steps_per_env = num_steps_per_env
        self.max_iterations = max_iterations
        self.empirical_normalization = empirical_normalization
        self.policy = policy
        self.algorithm = algorithm
        self.save_interval = save_interval
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.logger = logger
        self.neptune_project = neptune_project
        self.wandb_project = wandb_project
        self.resume = resume
        self.load_run = load_run
        self.load_checkpoint = load_checkpoint

    @classmethod
    def from_dict(cls, data: dict):
        """Create an instance from a dictionary."""
        return cls(
            seed=data.get("seed", 42),
            device=data.get("device", "cuda:0"),
            num_steps_per_env=data.get("num_steps_per_env"),
            max_iterations=data.get("max_iterations"),
            empirical_normalization=data.get("empirical_normalization"),
            policy=data.get("policy"),
            algorithm=data.get("algorithm"),
            save_interval=data.get("save_interval"),
            experiment_name=data.get("experiment_name"),
            run_name=data.get("run_name", ""),
            logger=data.get("logger", "tensorboard"),
            neptune_project=data.get("neptune_project", "isaaclab"),
            wandb_project=data.get("wandb_project", "isaaclab"),
            resume=data.get("resume", False),
            load_run=data.get("load_run", ".*"),
            load_checkpoint=data.get("load_checkpoint", "model_.*.pt")
        )

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {
            "seed": self.seed,
            "device": self.device,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "empirical_normalization": self.empirical_normalization,
            "policy": self.policy,
            "algorithm": self.algorithm,
            "save_interval": self.save_interval,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "logger": self.logger,
            "neptune_project": self.neptune_project,
            "wandb_project": self.wandb_project,
            "resume": self.resume,
            "load_run": self.load_run,
            "load_checkpoint": self.load_checkpoint
        }

    def copy(self):
        """Return a new instance with the same attributes."""
        return RslRlOnPolicyRunnerCfg(
            seed=self.seed,
            device=self.device,
            num_steps_per_env=self.num_steps_per_env,
            max_iterations=self.max_iterations,
            empirical_normalization=self.empirical_normalization,
            policy=self.policy,
            algorithm=self.algorithm,
            save_interval=self.save_interval,
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            logger=self.logger,
            neptune_project=self.neptune_project,
            wandb_project=self.wandb_project,
            resume=self.resume,
            load_run=self.load_run,
            load_checkpoint=self.load_checkpoint
        )