from models.cnn import CNN
from models.mlp import MLP
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from tqdm import tqdm
import math
import random
import torch
import torch.nn as nn
import os

class Agent():
    def __init__(self, env, eval_env, config, run_id):
        self.env = env
        self.env.reset(seed=run_id)
        self.env.action_space.seed(run_id)
        self.eval_env = eval_env
        self.eval_env.reset(seed=run_id)
        self.eval_env.action_space.seed(run_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.run_id = run_id
        self.eps = config.max_eps
        self.eps_step = (config.max_eps - config.min_eps) / config.n_eps
        self.max_reward = -math.inf

        if config.model == "mlp":
            self.policy_model = MLP(
                in_features=self.env.observation_space.shape[0]*4,
                out_features=self.env.action_space.n,
            ).to(self.device)
            self.target_model = MLP(
                in_features=self.env.observation_space.shape[0]*4,
                out_features=self.env.action_space.n,
            ).to(self.device)
        elif config.model == "cnn":
            self.policy_model = CNN(
                output_units=self.env.action_space.n,
            ).to(self.device)
            self.target_model = CNN(
                output_units=self.env.action_space.n,
            ).to(self.device)
        else:
            raise Exception(f"Invalid model: {config.model}")

        self.optimizer = torch.optim.Adam(
            params=self.policy_model.parameters(),
            lr=config.max_lr,
        )
        if hasattr(config, 'lr_decay'):
            self.lr_schedule = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=1000,
                gamma=config.lr_decay,
            )
        else:
            self.lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=config.min_lr/config.max_lr,
                total_iters=config.n_lr/config.training_freq,
            )

        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.save_model("model-initial.pt")

        self.replay_buffer = ReplayBuffer(
            config.buffer_size,
            self.env.observation_space.shape,
            self.device,
        )

        self.training_logger = Logger(
            f"results/{config.exp_id}/{run_id}/logs/training")
        self.eval_logger = Logger(
            f"results/{config.exp_id}/{run_id}/logs/eval",
        )
        self.t = 0

    def train(self):
        for i in tqdm(range(self.config.num_episodes_train), desc=f"Run {self.run_id}"):
            total_reward = 0
            episode_len = 0
            obs, _ = self.env.reset()
            done = False
            while not done:
                state = self.replay_buffer.get_state_for_new_obs(obs)
                if (random.random() < self.eps or 
                    self.t < self.config.learning_start):

                    action = self.env.action_space.sample()
                else:
                    self.policy_model.eval()
                    with torch.no_grad():
                        q = self.policy_model(torch.unsqueeze(state, dim=0))[0]
                    action = torch.argmax(q, dim=0).item()
                    self.training_logger.add_step_stats("q_a", q[action].item())

                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.add_action(action)
                self.replay_buffer.add_reward(reward)
                self.replay_buffer.add_done(done)

                if self.t >= self.config.learning_start and self.t % self.config.training_freq == 0:
                    self.train_step()

                self.eps = max(self.eps - self.eps_step, self.config.min_eps)
                self.training_logger.add_step_stats("eps", self.eps)
                self.training_logger.add_step_stats("lr", self.lr_schedule.get_last_lr()[0])

                total_reward += reward
                episode_len += 1
                self.t += 1
            
            if i % self.config.eval_freq == 0:
                self.eval()

            self.training_logger.add_episode_stats("training_reward", total_reward)
            self.training_logger.add_episode_stats("training_episode_len", episode_len)
            self.training_logger.flush(self.t)

    def train_step(self):
        states, actions, rewards, dones, next_states = self.replay_buffer.sample(self.config.batch_size)

        self.target_model.eval()
        with torch.no_grad():
            tq = self.target_model(next_states)
        tq_max, _ = torch.max(tq, dim=1) 
        if self.config.episodic:
            tq_max *= 1 - dones.int()
        targets = rewards + self.config.gamma * tq_max

        self.policy_model.train()
        q = self.policy_model(states)
        q_a = torch.gather(q, 1, actions.unsqueeze(dim=1)).squeeze(dim=1)

        loss = nn.MSELoss()(q_a, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_schedule.step()

        if self.t % self.config.target_update_freq == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())

        if self.t % self.config.model_save_freq == 0:
            self.save_model(f"model-checkpoint-{self.t}.pt")

        grad_norm = 0
        for p in self.policy_model.parameters():
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        self.training_logger.add_step_stats("grad_norm", grad_norm)
        self.training_logger.add_step_stats("loss", loss.item())
        
    def eval(self):
        total_reward = 0
        episode_len = 0
        buffer = ReplayBuffer(
            400,
            self.eval_env.observation_space.shape,
            self.device,
        )
        obs, _ = self.eval_env.reset()
        done = False
        while not done:
            state = buffer.get_state_for_new_obs(obs)
            self.policy_model.eval()
            with torch.no_grad():
                q = self.policy_model(torch.unsqueeze(state, dim=0))[0]
            action = torch.argmax(q, dim=0).item()

            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            done = terminated or truncated
            buffer.add_done(done)
            total_reward += reward
            episode_len += 1

        if total_reward > self.max_reward:
            self.max_reward = total_reward
            self.save_model(f"model-record-{total_reward}-{self.t}.pt")

        self.eval_logger.add_episode_stats("eval_reward", total_reward)
        self.eval_logger.add_episode_stats("eval_episode_len", episode_len)
        self.eval_logger.flush(self.t)

    def save_model(self, model_name):
        path = f"results/{self.config.exp_id}/{self.run_id}/models"
        if not os.path.exists(path):
            os.makedirs(path)

        self.policy_model.eval()
        torch.save(self.policy_model.state_dict(), f"{path}/{model_name}")