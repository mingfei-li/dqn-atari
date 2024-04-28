from config import Config
from rl_players import RLPlayer
import gymnasium as gym
from gymnasium.experimental.wrappers import RecordVideoV0
from pong_wrapper import PongWrapper
from tqdm import tqdm
import cProfile

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
    config.log_actions_freq  = 1
    config.log_training_freq  = 1
    config.eval_freq = 500
    play(config, debug=True)

def train():
    play(Config(), debug=False)

def play(config: Config, debug):
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env = PongWrapper(env, skip_frame=config.skip_frame)
    env = RecordVideoV0(env, config.record_path)

    player = RLPlayer(env, config, debug)
    done = True
    for _ in tqdm(range(config.nsteps_train), desc="Global step: "):
        if done:
            obs, _ = env.reset()

        action = player.get_action(obs)
        obs, reward, terminated, truncated, *_ = env.step(action)
        done = terminated or truncated
        player.update(action, reward, done)
    env.close()

if __name__ == "__main__":
    train()
    # cProfile.run("train()", "perf_stats_training.log")