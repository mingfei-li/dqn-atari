class Config():
    n_steps_train = 20_000_000
    max_eps = 1
    min_eps = 0.01
    n_eps = 2_000_000
    eps_schedule = 'exponential'
    max_lr = 7.5e-5
    min_lr = 7.5e-6
    n_lr = 5_000_000
    target_update_freq = 10_000
    batch_size = 96
    gamma = 0.99
    learning_start = 100_000
    buffer_size = 1_000_000
    eval_freq = 10_000
    episodic = True
    training_freq = 4
    model_save_freq = 1_000_000
    game = "breakout"
    exp_id = "exp-43"