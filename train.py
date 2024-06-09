from agent import Agent
from config import Config
import numpy as np
import random
import sys
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
    if len(sys.argv) != 3:
        print("Usage: python train.py <game> <number of training steps (in millions)>")
        sys.exit(1)
    game = sys.argv[1].capitalize()
    n_million_training_steps = int(sys.argv[2])
    config = Config()

    for run_id in [0, 42, 1234, 9999, 11111]:
        set_seed(run_id)
        Agent(
            config,
            game,
            run_id,
            n_million_training_steps * 1_000_000,
        ).train()