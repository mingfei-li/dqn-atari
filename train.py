from agent import Agent
from config import PongConfig
from utils.pong_wrapper import PongWrapper
import gymnasium as gym
import numpy as np
import random
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    env = gym.make('PongNoFrameskip-v4', render_mode="rgb_array", obs_type="grayscale")
    env = PongWrapper(env)
    config = PongConfig()
    for run_id in [0, 42, 1234, 9999, 11111]:
        set_seed(run_id)
        agent = Agent(env, config, run_id)
        agent.train()

    env.close()
