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

def play_pong_test():
    config = Config(
        mini_batch_size=3,
        replay_memory_size=10,
        target_netwrok_update_frequency=10,
        final_exploration_frame=100,
        replay_start_size=5,
        model_saving_frequency=100,
        initial_lr=0.01,
        final_lr=0.001,
        lr_anneal_steps=10,
    )
    play_pong(config, 1, debug=True)

def play_pong_training():
    play_pong(Config(), 50_000, debug=False)

def play_pong(config, episodes_to_train, debug):
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessingV0(env)
    env = RecordVideoV0(env, './videos')

    player = RLPlayer(env, config, debug)
    play(player, episodes_to_train)
    env.close()

def play(player, episodes_to_train):
    global_steps = 0
    global_start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for episode in tqdm(range(episodes_to_train), desc="Playing Eposide: "):
        start_time = time.time()
        steps = 0
        total_action_value = 0
        total_reward = 0
        total_loss = 0
        total_lr = 0
        explorations = 0

        while True:
            action, action_value, reward, terminated, loss, lr = player.step()
            global_steps += 1
            steps += 1
            elapsed_time = time.time() - start_time
            total_reward += reward
            if action_value is not None:
                total_action_value += action_value
            else:
                explorations += 1
            if loss is not None:
                total_loss += loss
            if lr is not None:
                total_lr += lr

            if terminated:
                tqdm.write(f"Episode {episode:6d}, global_steps {global_steps: 10d}: "
                        f"step = {steps: 6d}, "
                        f"elapsed_time = {str(timedelta(seconds=elapsed_time))}, "
                        f"avg_action_value = {total_action_value / max(steps-explorations, 1): 10.6f}, "
                        f"exploration_rate = {float(explorations) / steps: 10.6f}, "
                        f"reward = {total_reward: 5.02f}, "
                        f"avg_loss = {total_loss/steps: 10.6f}, "
                        f"avg_lr = {total_lr/steps: 10.6f}")

                with open(f'logs/training_log-{global_start_time}.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode,
                        steps,
                        elapsed_time,
                        total_action_value / max(steps-explorations, 1),
                        float(explorations) / steps,
                        total_reward,
                        total_loss / steps,
                        total_lr / steps,
                    ])
                player.reset()
                break
        
if __name__ == "__main__":
    cProfile.run("play_pong_training()", "logs/perf_stats_training.log")