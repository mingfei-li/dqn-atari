from collections import deque
from config import Config
from logger import Logger
from mlp_model import MLPModel
from tqdm import tqdm
import gymnasium as gym
import random
import torch
import torch.nn as nn
import os

class ReplayBuffer():
    def __init__(self, maxlen):
        self.obs = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)
    
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
        state = torch.zeros((4,) + self.obs[i].shape)
        state[3] = torch.tensor(self.obs[i])
        for j in range(1, 4):
            if i-j < 0:
                break
            if self.dones[i-j]:
                break
            state[3-j] = torch.tensor(self.obs[i-j])
        return state

class Agent():
    def __init__(self, env, config: Config, run_id):
        self.env = env
        self.config = config
        self.run_id = run_id
        self.eps = config.max_eps
        self.eps_step = (config.max_eps - config.min_eps) / config.n_eps
        self.lr = config.max_lr
        self.lr_step = (config.max_lr - config.min_lr) / config.n_lr

        self.policy_model = MLPModel(
            in_features=self.env.observation_space.shape[0]*4,
            out_features=self.env.action_space.n,
        )
        self.target_model = MLPModel(
            in_features=self.env.observation_space.shape[0]*4,
            out_features=self.env.action_space.n,
        )
        self.target_model.load_state_dict(self.policy_model.state_dict())

        self.replay_buffer = ReplayBuffer(maxlen=config.buffer_size)

        self.training_logger = Logger(
            f"results/logs/{config.exp_id}/{run_id}/training")
        self.testing_logger = Logger(
            f"results/logs/{config.exp_id}/{run_id}/testing",
        )
        self.t = 0

    def train(self):
        total_reward = 0
        obs, _ = self.env.reset()
        done = False
        while not done:
            self.replay_buffer.add_obs(obs)
            if (random.random() < self.eps or 
                self.t < self.config.learning_start):

                action = self.env.action_space.sample()
            else:
                self.policy_model.eval()
                state = self.replay_buffer.get_last_state()
                with torch.no_grad():
                    q = self.policy_model(torch.unsqueeze(state, dim=0))[0]
                action = torch.argmax(q, dim=0).item()
                self.training_logger.add_step_stats("q_a", q[action].item())

            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add_action(action)
            self.replay_buffer.add_reward(reward)
            self.replay_buffer.add_done(done)

            if self.t >= self.config.learning_start:
                self.train_step()

            self.eps = max(self.eps - self.eps_step, self.config.min_eps)
            self.lr = max(self.lr - self.lr_step, self.config.min_lr)
            self.training_logger.add_step_stats("eps", self.eps)
            self.training_logger.add_step_stats("lr", self.lr)

            total_reward += reward
            self.t += 1

        self.training_logger.add_episode_stats("training_reward", total_reward)
        self.training_logger.flush(self.t)

    def train_step(self):
        states, actions, rewards, dones, next_states = self.sample()

        self.target_model.eval()
        with torch.no_grad():
            tq = self.target_model(next_states)
        tq_max, _ = torch.max(tq, dim=1) 
        tq_max *= 1 - dones.int()
        targets = rewards + self.config.gamma * tq_max

        self.policy_model.train()
        q = self.policy_model(states)
        q_a = torch.gather(q, 1, actions.unsqueeze(dim=1)).squeeze(dim=1)

        loss = nn.MSELoss()(q_a, targets)
        optimizer = torch.optim.Adam(
            params=self.policy_model.parameters(),
            lr=self.lr,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.t % self.config.target_update_freq == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())

        grad_norm = 0
        for p in self.policy_model.parameters():
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        self.training_logger.add_step_stats("grad_norm", grad_norm)
        self.training_logger.add_step_stats("loss", loss.item())
        
    def sample(self):
        states, actions, rewards, dones, next_states = self.replay_buffer.sample(self.config.batch_size)

        return [
            torch.stack(states, dim=0),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.tensor(dones),
            torch.stack(next_states, dim=0),
        ]

    def test(self):
        total_reward = 0
        buffer = ReplayBuffer(4)
        obs, _ = self.env.reset()
        done = False
        while not done:
            buffer.add_obs(obs)
            state = buffer.get_last_state()
            self.policy_model.eval()
            with torch.no_grad():
                q = self.policy_model(torch.unsqueeze(state, dim=0))[0]
            action = torch.argmax(q, dim=0).item()

            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            buffer.add_done(done)
            total_reward += reward
        self.testing_logger.add_episode_stats("testing_reward", total_reward)
        self.testing_logger.flush(self.t)
        return total_reward

    def save_model(self, model_name):
        path = f"results/models/{self.config.exp_id}/{self.run_id}"
        if not os.path.exists(path):
            os.makedirs(path)

        self.policy_model.eval()
        torch.save(self.policy_model.state_dict(), f"{path}/{model_name}")

if __name__ == "__main__":
    for run_id in range(5):
        env = gym.make('CartPole-v0', render_mode="rgb_array")
        config = Config()
        agent = Agent(env, config, run_id)

        max_reward = 0
        for i in tqdm(range(config.num_episodes_train), desc=f"Run {run_id}"):
            agent.train()
            reward = agent.test()

            if reward >= max_reward:
                max_reward = reward
                agent.save_model(f"model-{reward}.pt")

        env.close()