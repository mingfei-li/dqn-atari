class Config():
    # output config
    output_path = "results/"
    model_path = output_path + "models/"
    log_path = output_path + "logs/"
    record_path = output_path + "videos/"

    # model and training config
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 1
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 5000000
    batch_size         = 32
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr_begin           = 0.00025
    lr_end             = 0.00025
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 1000000
    learning_start     = 50000
    obs_scale          = 255