from collections import deque
import gymnasium as gym
import numpy as np

class AtariWrapper(gym.Wrapper):
    def __init__(self, env, skip_frame=4):
        super().__init__(env)
        self.skip_frame = skip_frame
        self.obs_buffer = deque(maxlen=2)
        self.observation_space = gym.spaces.Box(0, 1, shape=(84,84), dtype=np.float32)
        self.episode_length = 0
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.skip_frame):
            obs, reward, terminated, truncated, _ = self.env.step(action)
            self.obs_buffer.append(self.transform(obs))
            total_reward += reward
            self.episode_length += 1

            if self.episode_length >= 50_000:
                truncated = True
            if terminated or truncated:
                break

        obs = np.max(np.stack(self.obs_buffer), axis=0)
        return obs, total_reward, terminated, truncated, {}
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.transform(obs)
        self.obs_buffer.clear()
        self.obs_buffer.append(obs)
        self.episode_length = 0
        return obs, info
    
    def transform(self, obs):
        # downsample by 2x
        obs = obs[::2, ::2]
        # crop
        obs = obs[15:99, :]
        # pad to 84x84
        obs = np.pad(obs, ((0, 0), (2, 2)), 'constant', constant_values=0)
        return obs / 255.0