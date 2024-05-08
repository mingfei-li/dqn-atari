class Config():
    pass

class CartPoleConfig(Config):
    num_episodes_train = 3000
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
    model = "mlp"
    exp_id = "exp-18,cartpole-v0,buffer=tensor"

class AtariPongConfig(Config):
    num_episodes_train = 5_000
    max_eps = 1
    min_eps = 0.1
    n_eps = 1_000_000
    max_lr = 0.0025
    min_lr = 0.000025
    n_lr = 2_500_000
    target_update_freq = 10_000
    batch_size = 128
    gamma = 0.99
    learning_start = 100_000
    buffer_size = 1_000_000
    test_freq = 10
    model = "conv_net"
    exp_id = "exp-33:atari-pong,buf_size=1m,batch_size=128,learning_start=100k,n_eps=1m,n_train=5k,lr=2.5e-4->2.5e-5,adam"