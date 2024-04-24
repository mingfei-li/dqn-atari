class Config(object):
    def __init__(self, **kwargs):
        # Hyper-parameters: default parameters are from DeepMind Nature paper
        self.mini_batch_size = kwargs.get("mini_batch_size", 32)
        self.replay_memory_size = kwargs.get("replay_memory_size", 1_000_000)
        self.agent_history_length = kwargs.get("agent_history_length", 4)
        self.target_netwrok_update_frequency = kwargs.get("target_netwrok_update_frequency", 10_000)
        self.discount_factor = kwargs.get("discount_factor", 0.99)
        self.action_repeat = kwargs.get("action_repeat", 4)
        self.update_frequency = kwargs.get("update_frequency", 4)
        self.initial_lr = kwargs.get("initial_lr", 0.00025)
        self.gradient_momentum = kwargs.get("gradient_momentum", 0.95)
        self.squared_gradient_momentum = kwargs.get("squared_gradient_momentum", 0.95)
        self.min_squared_gradient = kwargs.get("min_squared_gradient", 0.01)
        self.initial_exploration = kwargs.get("initial_exploration", 1)
        self.final_exploration = kwargs.get("final_exploration", 0.1)
        self.final_exploration_frame = kwargs.get("final_exploration_frame", 1_000_000)
        self.replay_start_size = kwargs.get("replay_start_size", 50_000)
        self.no_op_max = kwargs.get("no_op_max", 30)
        self.model_saving_frequency = kwargs.get("model_saving_frequency", 1_000_000)
        self.final_lr = kwargs.get("final_lr", 0.0005)
        self.lr_anneal_steps = kwargs.get("lr_anneal_steps", 2_5000_000)
        self.grad_norm_clip = kwargs.get("grad_norm_clip", 10)

        self.eps_anneal_rate = float(self.final_exploration - self.initial_exploration) / self.final_exploration_frame
        self.lr_anneal_rate = (self.final_lr - self.initial_lr) / self.lr_anneal_steps