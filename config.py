class PongConfig():
    num_episodes_train = 5_000
    max_eps = 1
    min_eps = 0.1
    n_eps = 5_000_000
    initial_lr = 1e-4
    lr_half_life = 1_000_000
    target_update_freq = 1_000
    batch_size = 96
    gamma = 0.99
    learning_start = 100_000
    buffer_size = 1_000_000
    eval_freq = 10
    episodic = True
    training_freq = 4
    model_save_freq = 1_000_000
    exp_id = "exp-8"