from config import Config
from gymnasium.experimental.wrappers import RecordVideoV0
from mlp_model import MLPModel
import gymnasium as gym
import torch
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python play.py <filename>")
        sys.exit(1)
    model_path = sys.argv[1]

    env = gym.make('CartPole-v0', render_mode="rgb_array")
    env = RecordVideoV0(env, video_folder="results/videos")

    model = MLPModel(
        in_features=env.observation_space.shape[0],
        out_features=env.action_space.n,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        state = torch.unsqueeze(torch.tensor(obs), dim=0)
        with torch.no_grad():
            q = model(state)[0]
        action = torch.argmax(q, dim=0).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    env.close()
    print(f"Reward: {total_reward}")
