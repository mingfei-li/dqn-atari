from collections import deque
import torch
import random

class ReplayBuffer():
    def __init__(self, maxlen, shape, device):
        self.maxlen = maxlen
        self.back = -1
        self.obs = torch.zeros((maxlen,) + shape, dtype=float, device=device)
        self.actions = torch.zeros(maxlen, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(maxlen, dtype=torch.float, device=device)
        self.dones = torch.zeros(maxlen, dtype=torch.bool, device=device)
        self.device = device
    
    def get_state_for_new_obs(self, obs):
        self.back += 1
        if self.back >= self.maxlen:
            raise Exception("Replay buffer full!")

        self.obs[self.back] = torch.tensor(obs, dtype=float, device=self.device)
        return self._get_state(self.back)

    def add_action(self, action):
        self.actions[self.back] = action
    
    def add_reward(self, reward):
        self.rewards[self.back] = reward
    
    def add_done(self, done):
        self.dones[self.back] = done

    def sample(self, size):
        assert self.back >= size
        indexes = random.sample(range(self.back), size)
        states = torch.stack([self._get_state(i) for i in indexes])
        actions = torch.tensor([self.actions[i] for i in indexes], device=self.device)
        rewards = torch.tensor([self.rewards[i] for i in indexes], device=self.device)
        dones = torch.tensor([self.dones[i] for i in indexes], device=self.device)
        next_states = torch.stack([self._get_state(i+1) for i in indexes])
        return states, actions, rewards, dones, next_states

    def _get_state(self, i):
        state = torch.zeros((4,) + self.obs[i].shape, device=self.device)
        state[3] = self.obs[i]
        for j in range(1, 4):
            if i-j < 0 or self.dones[i-j]:
                break
            state[3-j] = self.obs[i-j]
        return state