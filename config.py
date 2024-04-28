class Config():
    # output config
    output_path = "results/"
    model_path = output_path + "models/"
    log_path = output_path + "logs/"
    record_path = output_path + "videos/"

    # model and training config
    num_episodes_test = 50
    grad_clip         = False
    clip_val          = 10
    saving_freq       = 250000

    eval_freq         = 250_000
    record_freq       = 250_000
    soft_epsilon      = 0.05

    # logging config
    log_freq          = 1
    log_actions_freq  = 1000
    n_actions_log     = 10
    log_training_freq = 1000
    log_scalar_freq   = 1000
    log_histogram_freq = 5000
    log_window        = 20000
    log_image_freq    = 500

    # nature paper hyper params
    nsteps_train       = 5_000_000
    batch_size         = 32
    buffer_size        = 1_000_000
    target_update_freq = 10_000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = 1_000_000
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 1_000_000
    learning_start     = 50_000
    obs_scale          = 255