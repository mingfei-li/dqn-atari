from conv_net import ConvNet
from gymnasium.experimental.wrappers import AtariPreprocessingV0, RecordVideoV0
from mlp_model import MLPModel
from pong_wrapper import AtariWrapper
from replay_buffer_deque import ReplayBuffer
import gymnasium as gym
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

def cartpole(device, model_path, video_path):
    env = gym.make('CartPole-v0', render_mode="rgb_array")
    env = RecordVideoV0(env, video_folder=video_path)

    model = MLPModel(
        in_features=env.observation_space.shape[0]*4,
        out_features=env.action_space.n,
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    play(env, model)
    env.close()

def pong(device, model_path, video_path):
    env = gym.make('PongNoFrameskip-v4', render_mode="rgb_array", obs_type="grayscale")
    env = AtariWrapper(env)
    env = RecordVideoV0(env, video_folder=video_path)

    model = ConvNet(
        output_units=env.action_space.n,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    play(env, model)
    env.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python play.py <gamename> <filename>")
        sys.exit(1)
    model_path = sys.argv[2]

    base_dir = os.path.dirname(os.path.dirname(model_path))
    model_name = os.path.basename(model_path)
    video_path = os.path.join(base_dir, 'videos', model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    globals()[sys.argv[1]](device, model_path, video_path)
