from rl_players import LinearModelPongPlayer, LinearModelTestEnvPlayer, CNNModelPongPlayer
from utils.test_env import EnvTest
import csv
import gymnasium as gym
import time
import logging
from datetime import timedelta
from gymnasium.experimental.wrappers import RecordVideoV0, AtariPreprocessingV0
from tqdm import tqdm

def play_pong_test():
    config = {
        "mini_batch_size": 3,
        "experience_buffer_size": 10,
        "target_model_update_interval": 10,
        "gamma": 0.99,
        "training_frequency": 4,
        "learning_rate": 0.00025,
        "anneal_steps": 100,
        "replay_start_size": 5,
        "logging_level": logging.DEBUG,
    }
    play_pong(config, 50_000)

def play_pong_training():
    config = {
        "mini_batch_size": 32,
        "experience_buffer_size": 1_000_000,
        "target_model_update_interval": 10_000,
        "gamma": 0.99,
        "training_frequency": 4,
        "learning_rate": 0.00025,
        "anneal_steps": 1_000_000,
        "replay_start_size": 50_000,
        "logging_level": logging.INFO,
    }
    play_pong(config, 50_000)

def play_pong(config, episodes_to_train):
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessingV0(env, scale_obs=True)
    env = RecordVideoV0(env, './videos')

    player = CNNModelPongPlayer(env, config)
    play(player, episodes_to_train)
    env.close()

def play_test_env():
    env = EnvTest((3, 5))
    player = LinearModelTestEnvPlayer(env)
    play(player)

def play(player, episodes_to_train):
    for episode in tqdm(range(episodes_to_train), desc="Playing Eposide: "):
        start_time = time.time()
        steps = 0
        total_action_value = 0
        total_reward = 0
        total_loss = 0

        while True:
            action, action_value, reward, terminated, loss = player.step()
            steps += 1
            elapsed_time = time.time() - start_time
            total_reward += reward
            if action_value is not None:
                total_action_value += action_value
            if loss is not None:
                total_loss += loss
            tqdm.write(f"Episode {episode:6d}: "
                       f"step = {steps: 6d}, "
                       f"elapsed_time = {str(timedelta(seconds=elapsed_time))}, "
                       f"avg_action_value = {total_action_value/steps:20.15f}, "
                       f"reward = {total_reward:5.02f}, "
                       f"avg_loss = {total_loss/steps:20.15f}")
            if terminated:
                with open('training_log.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode,
                        steps,
                        elapsed_time,
                        total_action_value / steps,
                        total_reward,
                        total_loss / steps
                    ])
                player.reset()
                break

if __name__ == "__main__":
    # play_test_env()
    play_pong_test()