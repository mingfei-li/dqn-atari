class CartPoleConfig():
    num_episodes_train = 10_000
    max_eps = 0.05
    min_eps = 0.05
    n_eps = 50_000
    max_lr = 1e-4
    min_lr = 1e-4
    n_lr = 500_000
    target_update_freq = 50
    batch_size = 96
    gamma = 0.99
    learning_start = 20_000
    buffer_size = 50_000
    eval_freq = 1
    episodic = True
    training_freq = 1
    model_save_freq = 10_000
    model = "mlp"
    exp_id = "cartpole-v0:baseline"

class PongConfig():
    num_episodes_train = 8_000
    max_eps = 1
    min_eps = 0.1
    n_eps = 5_000_000
    max_lr = 7.5e-5
    min_lr = 7.5e-6
    n_lr = 2_500_000
    target_update_freq = 1_000
    batch_size = 128
    gamma = 0.99
    learning_start = 100_000
    buffer_size = 1_000_000
    eval_freq = 10
    episodic = False
    training_freq = 4
    model_save_freq = 1_000_000
    model = "cnn"
    exp_id = "pong:exp-seed42-18"
