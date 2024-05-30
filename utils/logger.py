from collections import deque
from torch.utils.tensorboard import SummaryWriter
from statistics import mean

class Logger():
    def __init__(self, log_dir):
        self.step_stats = {}
        self.episode_stats = {}
        self.writer = SummaryWriter(log_dir=f"{log_dir}")

    def add_step_stats(self, key, value):
        if key not in self.step_stats.keys():
            self.step_stats[key] = deque(maxlen=200)
        self.step_stats[key].append(value)

    def add_episode_stats(self, key, value):
        if key not in self.episode_stats.keys():
            self.episode_stats[key] = deque(maxlen=50)
        self.episode_stats[key].append(value)
    
    def flush(self, t):
        for key, value in self.step_stats.items():
            self.writer.add_scalar(key, mean(value), t)
        for key, value in self.episode_stats.items():
            self.writer.add_scalar(f"{key}.avg", mean(value), t)
            self.writer.add_scalar(f"{key}.max", max(value), t)
        self.writer.flush()