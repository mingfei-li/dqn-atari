from agent import Agent
from config import CartPoleConfig
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python play.py <filename>")
        sys.exit(1)
    cProfile.run(f"{sys.argv[1]}()", "perf_stats_training.log")