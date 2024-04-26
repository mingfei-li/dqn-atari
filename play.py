from config import Config
from episode import Episode
from rl_players import RLPlayer
import gymnasium as gym
from gymnasium.experimental.wrappers import RecordVideoV0
from tqdm import tqdm
from pong_wrapper import PongWrapper
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
    play(config, debug=True)

def train():
    play(Config(), debug=False)

def play(config: Config, debug):
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env = PongWrapper(env, skip_frame=config.skip_frame)
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
    train()
    # cProfile.run("train()", "perf_stats_training.log")