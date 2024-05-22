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
    env = gym.make('PongNoFrameskip-v4', render_mode="rgb_array", obs_type="grayscale")
    env = PongWrapper(env)
    config = PongConfig()
    set_seed(config.seed)
    agent = Agent(env, config)
    agent.train()
    env.close()
