from config import Config
import datetime
from rl_players import RLPlayer
from utils.test_env import EnvTest
import csv
import gymnasium as gym
import time
import logging
from datetime import timedelta
from gymnasium.experimental.wrappers import RecordVideoV0, AtariPreprocessingV0
from tqdm import tqdm
import cProfile
from statistics import mean, stdev
import os
from collections import Counter

class Episode():
    def __init__(self, id, config, n_actions):
        self.id = id
        self.t = 0
        self.actions = []
        self.q_list = []
        self.epsilons = []
        self.states = []
        self.obs_list = []
        self.rewards = []
        self.losses = []
        self.lrs = []
        self.training_info_list = []
        self.config = config
        self.done = False
        self.n_actions = n_actions

    def update_action(self, state, action, q, eps):
        self.states.append(state)
        self.actions.append(action)
        if q is not None:
            self.q_list.append(q)
        self.epsilons.append(eps)

    def update_feedback(self, obs, reward, done):
        self.obs_list.append(obs)
        self.rewards.append(reward)
        self.done = done
    
    def update_training(self, loss, lr, training_info):
        if loss is not None:
            self.losses.append(loss)
            self.lrs.append(lr)
            self.training_info_list.append(training_info)

    def log(self):
        self.log_progress()

    def log_progress(self):
        action_values = [self.q_list[i][self.actions[i]].item() for i in range(len(self.q_list))]
        avg_action_value = 0 if len(action_values) == 0 else mean(action_values)
        stdev_action_value = 0 if len(action_values) == 0 else stdev(action_values)
        avg_eps = mean(self.epsilons)
        total_reward = sum(self.rewards)

        avg_loss = 0 if len(self.losses) == 0 else mean(self.losses)
        avg_lr = 0 if len(self.lrs) == 0 else mean(self.lrs)

        action_distr = {k: v / len(self.actions) for k, v in Counter(self.actions).items()}
        action_distr_log = "["
        for i in range(self.n_actions):
            if i > 0:
                action_distr_log += ", "
            action_distr_log += f"{i}: {action_distr.get(i, 0):6.2%}"
        action_distr_log += "]"

        tqdm.write(f"Episode {self.id:6d} | "
                    f"t = {self.t: 4d} | "
                    f"avg_q = {avg_action_value:8.5f} | "
                    f"stdev_q = {stdev_action_value:8.5f} | "
                    f"avg_eps = {avg_eps:5.3f} | "
                    f"reward = {total_reward:2.0f} | "
                    f"avg_loss = {avg_loss:8.5f} | "
                    f"avg_lr = {avg_lr:8.5f} | "
                    f"action_distribution = {action_distr_log}")

        if not os.path.exists(self.config.log_path):
            os.makedirs(self.config.log_path)
        with open(self.config.log_path + f'/training_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.id,
                self.t,
                avg_action_value,
                stdev_action_value,
                avg_eps,
                total_reward,
                avg_loss,
                avg_lr,
            ])

def test():
    config = Config()
    config.batch_size = 3
    config.buffer_size = 10
    config.target_update_freq = 20
    config.eps_nsteps = 2000
    config.learning_start = 5
    config.learning_freq = 10
    config.saving_freq = 100
    config.lr_begin = 0.01
    config.lr_end = 0.001
    config.lr_nsteps = 2000
    config.nsteps_train = 5000
    play(config, debug=True)

def train():
    play(Config(), debug=False)

def play(config: Config, debug):
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessingV0(env, noop_max=0)
    env = RecordVideoV0(env, config.record_path)

    player = RLPlayer(env, config, debug)
    episode = None
    episode_id = 0
    for _ in tqdm(range(config.nsteps_train), desc="Global step: "):
        if episode is None or episode.done:
            if episode_id > 0:
                episode.log()
            episode_id += 1
            episode = Episode(episode_id, config, env.action_space.n)
            obs, _ = env.reset()

        state, action, q, eps = player.get_action(obs)
        obs, reward, terminated, truncated, *_ = env.step(action)
        done = terminated or truncated
        loss, lr, training_info = player.update(action, reward, done)

        episode.update_action(state, action, q, eps)
        episode.update_feedback(obs, reward, done)
        episode.update_training(loss, lr, training_info)
        episode.t += 1

    env.close()

if __name__ == "__main__":
    test()
    # cProfile.run("train()", "results/logs/perf_stats_training.log")