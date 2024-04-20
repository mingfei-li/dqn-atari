from rl_players import LinearModelPongPlayer, LinearModelTestEnvPlayer, CNNModelPongPlayer
from utils.test_env import EnvTest
import logging
import gymnasium as gym
from gymnasium.experimental.wrappers import RecordVideoV0
from tqdm import tqdm

def play_pong_test():
    config = {
        "replay_start_size": 5,
        "experience_buffer_size": 10,
        "mini_batch_size": 3,
        "gamma": 0.99,
        "learning_rate": 0.01,
        "anneal_steps": 100,
        "target_model_update_interval": 10,
        "device": "cpu"
    }
    play_pong(config)

def play_pong(config):
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = RecordVideoV0(env, './videos')

    player = CNNModelPongPlayer(env, config)
    play(player)
    env.close()

def play_test_env():
    env = EnvTest((3, 5))
    player = LinearModelTestEnvPlayer(env)
    play(player)

def play(player):
    for episode in tqdm(range(10), desc="Playing Eposide: "):
        steps = 0
        total_action_value = 0
        total_reward = 0
        total_loss = 0

        while True:
            action, action_value, reward, terminated, loss = player.step()
            steps += 1
            total_reward += reward
            if action_value is not None:
                total_action_value += action_value
            if loss is not None:
                total_loss += loss
            tqdm.write(f"Episode {episode:8d}, step {steps: 8d}: "
                       f"avg_action_value = {total_action_value/steps:10.02f}, "
                       f"reward={total_reward/steps:10.02f}, "
                       f"loss={total_loss/steps:10.02f}")
            if terminated:
                player.reset()
                break

if __name__ == "__main__":
    # play_test_env()
    play_pong_test()