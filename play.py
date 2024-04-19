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
    for step in tqdm(range(10000), desc="Playing"):
        action, action_value, reward, terminated, training_loss = player.step()
        tqdm.write(f"Step {step:5d}: action={action}, action_value={action_value}, reward={reward}, loss={training_loss}")

if __name__ == "__main__":
    play_test_env()
    # play_pong()