from rl_players import LinearModelPongPlayer, LinearModelTestEnvPlayer
from utils.test_env import EnvTest
import logging
import gymnasium as gym
from tqdm import tqdm

def play_pong():
    env = gym.make("ALE/Pong-v5", render_mode="human")
    player = LinearModelPongPlayer(env)
    play(player)
    env.close()

def play_test_env():
    env = EnvTest((3, 5))
    player = LinearModelTestEnvPlayer(env)
    play(player)

def play(player):
    for episode in tqdm(range(10000), desc="Playing Eposide: "):
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
            tqdm.write(f"Episode {episode:8d}, step {steps: 8d}: action_value={total_action_value:10.02f}, reward={total_reward:10.02f}, loss={total_loss:10.02f}")
            if terminated:
                player.reset()
                break

if __name__ == "__main__":
    play_test_env()
    # play_pong()