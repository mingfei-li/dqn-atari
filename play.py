from gymnasium.experimental.wrappers import RecordVideoV0
from gymnasium.experimental.wrappers import AtariPreprocessingV0
from models.cnn import CNN
from utils.replay_buffer import ReplayBuffer
import gymnasium as gym
import random
import torch
import os
import sys

def play(env, model):
    obs, _ = env.reset()
    total_reward = 0
    episode_len = 0
    buffer = ReplayBuffer(
        400,
        env.observation_space.shape,
        device,
    )
    done = False
    while not done:
        if random.random() < 0.01:
            action = env.action_space.sample()
        else:
            state = buffer.get_state_for_new_obs(obs)
            with torch.no_grad():
                q = model(torch.unsqueeze(state, dim=0))[0]
            action = torch.argmax(q, dim=0).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add_done(done)
        total_reward += reward
        episode_len += 1
    
    print(f"Reward: {total_reward}")
    print(f"Episode Length: {episode_len}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python play.py <game> <filename>")
        sys.exit(1)
    game = sys.argv[1]
    model_path = sys.argv[2]

    base_dir = os.path.dirname(os.path.dirname(model_path))
    model_name = os.path.basename(model_path)
    video_path = os.path.join(base_dir, 'videos', model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(
        f'{game.capitalize()}NoFrameskip-v4',
        render_mode="rgb_array",
    )
    env = AtariPreprocessingV0(env)
    env = RecordVideoV0(env, video_folder=video_path)

    model = CNN(
        output_units=env.action_space.n,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    play(env, model)
    env.close()
