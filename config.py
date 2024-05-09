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
    exp_id = "exp-20,cartpole-v0,buffer=tensor"

class AtariPongConfig(Config):
    num_episodes_train = 5_000
    max_eps = 1
    min_eps = 0.1
    n_eps = 2_000_000
    max_lr = 1e-4
    min_lr = 1e-4
    n_lr = 2_500_000
    target_update_freq = 10_000
    batch_size = 96
    gamma = 0.99
    learning_start = 50_000
    buffer_size = 1_500_000
    test_freq = 10
    model = "conv_net"
    exp_id = "exp-42:atari-pong,buf_size=1.5m,batch_size=96,learning_start=50k,lr=1e-4,n_train=5k,n_eps=2m,adam"

class EasyPongConfig(Config):
    num_episodes_train = 25_000
    max_eps = 1
    min_eps = 0.1
    n_eps = 1_000_000
    max_lr = 1e-4
    min_lr = 1e-4
    n_lr = 2_500_000
    target_update_freq = 10_000
    batch_size = 96
    gamma = 0.99
    learning_start = 50_000
    buffer_size = 1_500_000
    test_freq = 10
    model = "conv_net"
    exp_id = "easy-pong:exp-3:buf_size=1.5m,batch_size=96,learning_start=50k,lr=5e-4,n_train=25k,n_eps=1m,adam"