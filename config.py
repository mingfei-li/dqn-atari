class Config():
    num_episodes_train = 5000
    max_eps = 0.9
    min_eps = 0.9
    n_eps = 50_000
    max_lr = 1e-4
    min_lr = 1e-4
    n_lr = 500_000
    target_update_freq = 50
    batch_size = 96
    gamma = 0.99
    learning_start = 10_000
    buffer_size = 50_000
    grad_clip = 1e4
    exp_id = "batchsize=96,lr=1e-4,layer=6,eps=0.9,num_episodes_train=5000"