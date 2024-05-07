from config import Config
from gymnasium.experimental.wrappers import RecordVideoV0
from mlp_model import MLPModel
from replay_buffer_deque import ReplayBuffer
from pathlib import Path
import gymnasium as gym
import torch
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python play.py <filename>")
        sys.exit(1)
    model_path = sys.argv[1]
    path = Path(model_path)
    video_path = f"results/videos/{Path(*path.parts[2:])}"

    env = gym.make('CartPole-v0', render_mode="rgb_array")
    env = RecordVideoV0(env, video_folder=video_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPModel(
        in_features=env.observation_space.shape[0]*4,
        out_features=env.action_space.n,
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

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
        episode_len += episode_len
    
    env.close()
    print(f"Reward: {total_reward}")
    print(f"Episode Length: {episode_len}")
