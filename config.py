class Config():
    pass

class CartPoleConfig(Config):
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
    test_freq = 1
    clip_val = 0
    betas = (0.9, 0.999)
    episodic = True
    training_freq = 4
    model_save_freq = 10_000
    model = "mlp"
    exp_id = "cartpole-v0,exp-22,buffer=tensor,training_freq=4"

class AtariPongConfig(Config):
    num_episodes_train = 50_000
    max_eps = 1
    min_eps = 0.1
    n_eps = 5_000_000
    max_lr = 7.5e-5
    min_lr = 7.5e-6
    n_lr = 2_500_000
    target_update_freq = 1_000
    batch_size = 96
    gamma = 0.99
    learning_start = 100_000
    buffer_size = 1_000_000
    test_freq = 10
    clip_val = 0
    betas = (0.9, 0.999)
    episodic = False
    training_freq = 4
    model_save_freq = 1_000_000
    model = "conv_net"
    exp_id = "pong:exp-16"

class EasyPongConfig(Config):
    num_episodes_train = 100_000
    max_eps = 1
    min_eps = 0.1
    n_eps = 2_500_000
    max_lr = 7.5e-5
    min_lr = 7.5e-6
    n_lr = 2_500_000
    target_update_freq = 10_000
    batch_size = 96
    gamma = 0.99
    learning_start = 100_000
    buffer_size = 1_000_000
    test_freq = 500
    clip_val = 0
    betas = (0.9, 0.999)
    episodic = False
    training_freq = 4
    model_save_freq = 100_000
    model = "conv_net"
    exp_id = "easy-pong-v4:exp-6:buf_size=1m,batch_size=96,learning_start=100k,lr=7.5e-5->7.5e-6,n_train=100k,n_eps=2.5m,episodic=false,training_freq=4"