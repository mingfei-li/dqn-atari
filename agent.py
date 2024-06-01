from models.cnn import CNN
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from tqdm import tqdm
import math
import random
import torch
import torch.nn as nn
import os

class Agent():
    def __init__(self, env, config, run_id):
        self.env = env
        self.env.reset(seed=run_id)
        self.env.action_space.seed(run_id)
        self.run_id = run_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.eps = config.max_eps
        self.eps_step = (config.max_eps - config.min_eps) / config.n_eps
        self.max_reward = -math.inf

        self.policy_network = CNN(
            output_units=self.env.action_space.n,
        ).to(self.device)
        self.target_network = CNN(
            output_units=self.env.action_space.n,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            params=self.policy_network.parameters(),
            lr=config.initial_lr,
        )
        half_life_iters = config.lr_half_life / config.training_freq
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=0.5**(1.0/half_life_iters),
        )

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.save_model("model-initial.pt")

        self.replay_buffer = ReplayBuffer(
            config.buffer_size,
            self.env.observation_space.shape,
            self.device,
        )

        self.training_logger = Logger(
            f"results/{config.game}/{config.exp_id}/{run_id}/logs/training")
        self.eval_logger = Logger(
            f"results/{config.game}/{config.exp_id}/{run_id}/logs/eval",
        )

    def train(self):
        episode_reward = 0
        episode_len = 0
        obs, _ = self.env.reset()
        for t in tqdm(range(self.config.n_steps_train), desc=f"Run {self.run_id}"):
            state = self.replay_buffer.get_state_for_new_obs(obs)
            if random.random() < self.eps or t < self.config.learning_start:
                action = self.env.action_space.sample()
            else:
                self.policy_network.eval()
                with torch.no_grad():
                    q = self.policy_network(torch.unsqueeze(state, dim=0))[0]
                action = torch.argmax(q, dim=0).item()
                self.training_logger.add_step_stats("q_a", q[action].item())

            obs, reward, terminated, truncated, _ = self.env.step(action)
            # if reward != 0:
            #     reward = reward / abs(reward)
            done = terminated or truncated
            self.replay_buffer.add_action(action)
            self.replay_buffer.add_reward(reward)
            self.replay_buffer.add_done(done)

            if t >= self.config.learning_start and t % self.config.training_freq == 0:
                self.train_step(t)

            if t % self.config.eval_freq == 0:
                self.eval(t)

            self.eps = max(self.eps - self.eps_step, self.config.min_eps)
            self.training_logger.add_step_stats("eps", self.eps)
            episode_reward += reward
            episode_len += 1

            if done:
                self.training_logger.add_episode_stats(
                    "training_reward",
                    episode_reward,
                )
                self.training_logger.add_episode_stats(
                    "training_episode_len", 
                    episode_len,
                )
                self.training_logger.flush(t)
                episode_reward = 0
                episode_len = 0
                obs, _ = self.env.reset()
                done = False

    def train_step(self, t):
        states, actions, rewards, dones, next_states = (
            self.replay_buffer.sample(self.config.batch_size)
        )

        self.target_network.eval()
        with torch.no_grad():
            tq = self.target_network(next_states)
        tq_max, _ = torch.max(tq, dim=1) 
        if self.config.episodic:
            tq_max *= 1 - dones.int()
        targets = rewards + self.config.gamma * tq_max

        self.policy_network.train()
        q = self.policy_network(states)
        q_a = torch.gather(q, 1, actions.unsqueeze(dim=1)).squeeze(dim=1)

        loss = nn.MSELoss()(q_a, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        if t % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        if t % self.config.model_save_freq == 0:
            self.save_model(f"model-checkpoint-{t}.pt")

        grad_norm = 0
        for p in self.policy_network.parameters():
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        self.training_logger.add_step_stats("grad_norm", grad_norm)
        self.training_logger.add_step_stats("loss", loss.item())
        self.training_logger.add_step_stats(
            "lr",
            self.lr_scheduler.get_last_lr()[0],
        )
        
    def eval(self, t):
        episode_reward = 0
        episode_len = 0
        buffer = ReplayBuffer(
            10_000,
            self.env.observation_space.shape,
            self.device,
        )
        obs, _ = self.env.reset()
        done = False
        while not done:
            if random.random() < 0.01:
                action = self.env.action_space.sample()
            else:
                state = buffer.get_state_for_new_obs(obs)
                self.policy_network.eval()
                with torch.no_grad():
                    q = self.policy_network(torch.unsqueeze(state, dim=0))[0]
                action = torch.argmax(q, dim=0).item()

            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_len += 1

        if episode_reward > self.max_reward:
            self.max_reward = episode_reward
            self.save_model(f"model-record-{episode_reward}-{t}.pt")

        self.eval_logger.add_episode_stats("eval_reward", episode_reward)
        self.eval_logger.add_episode_stats("eval_episode_len", episode_len)
        self.eval_logger.flush(t)

    def save_model(self, model_name):
        path = f"results/{self.config.game}/{self.config.exp_id}/{self.run_id}/models"
        if not os.path.exists(path):
            os.makedirs(path)

        self.policy_network.eval()
        torch.save(self.policy_network.state_dict(), f"{path}/{model_name}")