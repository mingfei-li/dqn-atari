import torch
import random

class ReplayBuffer():
    def __init__(self, maxlen, shape, device):
        self.maxlen = maxlen
        self.len = 0
        self.back = -1
        self.obs = torch.zeros((maxlen,) + shape, dtype=torch.float, device=device)
        self.actions = torch.zeros(maxlen, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(maxlen, dtype=torch.float, device=device)
        self.dones = torch.zeros(maxlen, dtype=torch.bool, device=device)
        self.device = device
    
    def get_state_for_new_obs(self, obs):
        self.len += 1
        self.back = (self.back + 1) % self.maxlen
        self.obs[self.back] = torch.tensor(obs, dtype=float, device=self.device)
        return self._get_state(self.back)

    def add_action(self, action):
        self.actions[self.back] = action
    
    def add_reward(self, reward):
        self.rewards[self.back] = reward
    
    def add_done(self, done):
        self.dones[self.back] = done

    def sample(self, size):
        if self.len <= self.maxlen:
            indexes = random.sample(range(self.back), size)
        else:
            indexes = random.sample(range(self.back + 4, self.back + self.maxlen), size)
            indexes = [i % self.maxlen for i in indexes]
        states = torch.stack([self._get_state(i) for i in indexes])
        actions = torch.tensor([self.actions[i] for i in indexes], device=self.device)
        rewards = torch.tensor([self.rewards[i] for i in indexes], device=self.device)
        dones = torch.tensor([self.dones[i] for i in indexes], device=self.device)
        next_states = torch.stack([self._get_state(i+1) for i in indexes])
        return states, actions, rewards, dones, next_states

    def _get_state(self, i):
        if i >= 3 and not torch.any(self.dones[i-3:i]).item():
            return self.obs[i-3:i+1]

        state = torch.zeros((4,) + self.obs[i].shape, device=self.device)
        state[3] = self.obs[i]
        for j in range(1, 4):
            k = (i-j) % self.maxlen
            if self.dones[k]:
                break
            state[3-j] = self.obs[k]
        return state