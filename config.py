class Config():
    num_episodes_train = 2000
    max_eps = 0.05
    min_eps = 0.05
    n_eps = 50_000
    max_lr = 1e-4
    min_lr = 1e-4
    n_lr = 500_000
    target_update_freq = 50
    batch_size = 64
    gamma = 0.99
    learning_start = 10_000
    buffer_size = 50_000
    exp_id = "v0,batchsize=64,lr=1e-4,state=4frames"