from collections import deque
import torch
import random

class ReplayBuffer():
    def __init__(self, maxlen, device):
        self.obs = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)
        self.device = device
    
    def add_obs(self, obs):
        self.obs.append(obs)

    def add_action(self, action):
        self.actions.append(action)
    
    def add_reward(self, reward):
        self.rewards.append(reward)
    
    def add_done(self, done):
        self.dones.append(done)

    def get_last_state(self):
        return self._get_state(len(self.obs)-1)

    def sample(self, size):
        assert len(self.obs)-4 >= size
        indexes = random.sample(range(3, len(self.obs)-1), size)
        states = [self._get_state(i) for i in indexes]
        actions = [self.actions[i] for i in indexes]
        rewards = [self.rewards[i] for i in indexes]
        dones = [self.dones[i] for i in indexes]
        next_states = [self._get_state(i+1) for i in indexes]
        return states, actions, rewards, dones, next_states

    def _get_state(self, i):
        state = torch.zeros((4,) + self.obs[i].shape, device=self.device)
        state[3] = torch.tensor(self.obs[i], device=self.device)
        for j in range(1, 4):
            if i-j < 0:
                break
            if self.dones[i-j]:
                break
            state[3-j] = torch.tensor(self.obs[i-j], device=self.device)
        return state