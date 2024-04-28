from config import Config
from rl_players import RLPlayer
import gymnasium as gym
from gymnasium.experimental.wrappers import RecordVideoV0
from pong_wrapper import AtariWrapper
from tqdm import tqdm
import cProfile

def play():
    config = Config()
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array", obs_type="grayscale")
    env = AtariWrapper(env, skip_frame=config.skip_frame)
    env = RecordVideoV0(env, config.record_path)
    player = RLPlayer(env, config)
    for _ in tqdm(range(10_000), desc="Episode: "):
        done = False
        obs, _ = env.reset()
        while not done:
            action = player.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            player.update(action, reward, done)
    env.close()

if __name__ == "__main__":
    play()
    # cProfile.run("train()", "perf_stats_training.log")