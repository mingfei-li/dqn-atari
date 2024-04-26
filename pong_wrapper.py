import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as transforms

from collections import deque

class PongWrapper(gym.Wrapper):
    def __init__(self, env, skip_frame=4):
        super().__init__(env)
        self.skip_frame = skip_frame
        self.obs_buffer = deque(maxlen=2)
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.skip_frame):
            obs, reward, terminated, truncated, _ = self.env.step(action)
            self.obs_buffer.append(self.transform(obs))
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        obs = np.max(np.stack(self.obs_buffer), axis=0)
        return obs, total_reward, done, done, {}
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.transform(obs)
        self.obs_buffer.clear()
        self.obs_buffer.append(obs)
        return obs, info
    
    def transform(self, obs):
        frame = torch.tensor(obs).permute(2, 0, 1)

        transform = transforms.Compose([
             transforms.ToPILImage(),
             transforms.Resize((110, 84)),
             transforms.Grayscale(),
             transforms.ToTensor(),
        ])

        frame = transform(frame).squeeze(0)[17:101, :]
        return frame.numpy()
