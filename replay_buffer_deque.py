from collections import deque
import torch
import random

class ReplayBuffer():
    def __init__(self, maxlen, shape, device):
        self.shape = shape
        self.obs = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)
        self.device = device
    
    def get_state_for_new_obs(self, obs):
        self.obs.append(obs)
        return self._get_state(len(self.obs)-1)

    def add_action(self, action):
        self.actions.append(action)
    
    def add_reward(self, reward):
        self.rewards.append(reward)
    
    def add_done(self, done):
        self.dones.append(done)

    def sample(self, size):
        assert len(self.obs)-4 >= size
        indexes = random.sample(range(3, len(self.obs)-1), size)
        states = torch.stack([self._get_state(i) for i in indexes])
        actions = torch.tensor([self.actions[i] for i in indexes], device=self.device)
        rewards = torch.tensor([self.rewards[i] for i in indexes], device=self.device)
        dones = torch.tensor([self.dones[i] for i in indexes], device=self.device)
        next_states = torch.stack([self._get_state(i+1) for i in indexes])
        return states, actions, rewards, dones, next_states

    def _get_state(self, i):
        state = torch.zeros((4,) + self.shape, device=self.device)
        state[3] = torch.tensor(self.obs[i], device=self.device)
        for j in range(1, 4):
            if i-j < 0 or self.dones[i-j]:
                break
            state[3-j] = torch.tensor(self.obs[i-j], device=self.device)
        return state