from agent import Agent
from config import PongConfig
from utils.pong_wrapper import PongWrapper
import gymnasium as gym
import numpy as np
import random
import sys
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    run_id = 0
    set_seed(run_id)
    env = gym.make('PongNoFrameskip-v4', render_mode="rgb_array", obs_type="grayscale")
    env = PongWrapper(env)
    eval_env = gym.make('PongNoFrameskip-v4', render_mode="rgb_array", obs_type="grayscale")
    eval_env = PongWrapper(eval_env)
    config = PongConfig()
    agent = Agent(env, eval_env, config, run_id)
    agent.train()
    env.close()
