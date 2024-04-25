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
from statistics import mean
import os

def test():
    config = Config()
    config.batch_size = 3
    config.buffer_size = 10
    config.target_update_freq = 20
    config.eps_nsteps = 100
    config.learning_start = 5
    config.learning_freq = 10
    config.saving_freq = 100
    config.lr_begin = 0.01
    config.lr_end = 0.001
    config.lr_nsteps = 10
    play(config, 1, debug=True)

def train():
    play(Config(), 50_000, debug=False)

def play(config, episodes_to_train, debug):
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessingV0(env)
    env = RecordVideoV0(env, config.record_path)

    player = RLPlayer(env, config, debug)
    global_start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for episode in tqdm(range(episodes_to_train), desc="Playing Eposide: "):
        start_time = time.time()
        steps = 0

        rewards = []
        action_values = []
        losses = []
        lrs = []
        epsilons = []

        obs, _ = env.reset()
        player.reset(obs)

        while True:
            action, action_value, eps  = player.get_action()
            obs, reward, terminated, truncated, *_ = env.step(action)
            done = terminated or truncated
            loss, lr = player.update(action, obs, reward, done)

            steps += 1
            elapsed_time = time.time() - start_time

            epsilons.append(eps)
            rewards.append(reward)
            if action_value is not None:
                action_values.append(action_value)
            if loss is not None:
                losses.append(loss)
            if lr is not None:
                lrs.append(lr)

            if done:
                if len(action_values) > 0:
                    avg_action_value = mean(action_values)
                else:
                    avg_action_value = 0

                if len(lrs) > 0:
                    avg_lr = mean(lrs)
                    avg_loss = mean(losses)
                else:
                    avg_lr = 0
                    avg_loss = 0
                avg_eps = mean(epsilons)
                total_reward = sum(rewards)

                tqdm.write(f"Episode {episode:6d} | "
                           f"global_steps {player.t: 10d} | "
                           f"step = {steps: 6d} | "
                           f"elapsed_time = {str(timedelta(seconds=elapsed_time))} | "
                           f"avg_action_value = {avg_action_value: 10.6f} | "
                           f"exploration_rate = {avg_eps: 10.6f} | "
                           f"reward = {total_reward: 5.02f} | "
                           f"avg_loss = {avg_loss: 10.6f} | "
                           f"avg_lr = {avg_lr: 10.6f}")

                if not os.path.exists(config.log_path):
                    os.makedirs(config.log_path)
                with open(config.log_path + f'/training_log-{global_start_time}.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode,
                        steps,
                        elapsed_time,
                        avg_action_value, 1,
                        avg_eps,
                        total_reward,
                        avg_loss,
                        avg_lr,
                    ])
                break

    env.close()
        
if __name__ == "__main__":
    train()