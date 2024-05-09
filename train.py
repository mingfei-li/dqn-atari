from agent import Agent
from config import CartPoleConfig, AtariPongConfig, EasyPongConfig
from gymnasium.experimental.wrappers import AtariPreprocessingV0, GrayscaleObservationV0
from pong_wrapper import PongWrapper
import cProfile
import gymnasium as gym
import sys

def cartpole():
    for run_id in range(5):
        env = gym.make('CartPole-v0', render_mode="rgb_array")
        config = CartPoleConfig()
        agent = Agent(env, config, run_id)
        agent.train()
        env.close()

def atari_pong():
    for run_id in range(5):
        env = gym.make('PongNoFrameskip-v4', render_mode="rgb_array")
        env = AtariPreprocessingV0(env, scale_obs=True)
        config = AtariPongConfig()
        agent = Agent(env, config, run_id)
        agent.train()
        env.close()

def easy_pong():
    for run_id in range(5):
        env = gym.make('PongNoFrameskip-v4', render_mode="rgb_array", obs_type="grayscale")
        env = PongWrapper(env, training=True)
        config = EasyPongConfig()
        agent = Agent(env, config, run_id)
        agent.train()
        env.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python play.py <filename>")
        sys.exit(1)
    cProfile.run(f"{sys.argv[1]}()", "perf_stats_training.log")