from config import Config
from replay_buffer import ReplayBuffer
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboard_logger import TensorboardLogger
import os

class RLPlayer(object):
    def __init__(self, env, config: Config):
        self.env = env
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.q_net = ConvNet(self.env.action_space.n).to(self.device)
        self.target_q_net = ConvNet(self.env.action_space.n).to(self.device)
        self.update_target_net()

        self.logger = TensorboardLogger(config)

        self.replay_buffer = ReplayBuffer(
            n=self.config.buffer_size,
            state_history=self.config.state_history,
            scale=self.config.obs_scale,
            device=self.device,
        )

        self.t = 0
        self.eps = self.config.eps_begin
        self.esp_delta = (self.config.eps_end - self.config.eps_begin) / self.config.eps_nsteps
        self.lr = self.config.lr_begin
        self.lr_delta = (self.config.lr_end - self.config.lr_begin) / self.config.lr_nsteps

    def update_eps(self):
        self.eps += self.esp_delta
        if self.t >= self.config.eps_nsteps:
            self.eps = self.config.eps_end
    
    def update_lr(self):
        self.lr += self.lr_delta
        if self.t >= self.config.lr_nsteps:
            self.lr = self.config.lr_end

    def get_action(self, obs):
        self.t += 1

        self.replay_buffer.add_frame(obs)
        state = self.replay_buffer.get_last_state()

        self.target_q_net.eval()
        with torch.no_grad():
            q = self.target_q_net(torch.unsqueeze(state, dim=0))[0]

        if self.t < self.config.learning_start or random.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = torch.argmax(q, dim=0).item()

        self.logger.add_action_summary(action, obs, q)
        return action

    def update(self, action, reward, done):
        self.replay_buffer.add_action(action)
        self.replay_buffer.add_reward(reward)
        self.replay_buffer.add_done(done)

        if (self.t >= self.config.learning_start and 
            self.t % self.config.learning_freq == 0):

            self.train()

        self.update_eps()
        self.update_lr()

        self.logger.add_reward_summary(reward, done)
        self.logger.log(self.t, self.eps, self.lr, self.q_net, self.target_q_net)

    def train(self):
        s, a, r, d, ns = self.replay_buffer.sample(self.config.batch_size)
        s = s[~d]
        a = a[~d]
        r = r[~d]
        ns = ns[~d]

        if len(s) == 0:
            return

        self.target_q_net.eval()
        with torch.no_grad():
            tq = self.target_q_net(ns)
            tq_a, _ = torch.max(tq, dim=1)
            target = r + self.config.gamma * tq_a
        
        self.q_net.train()
        q = self.q_net(s)
        q_a = q[torch.arange(q.size(0)), a]
        loss = nn.HuberLoss()(q_a, target)

        optimizer = torch.optim.RMSprop(
            params=self.q_net.parameters(),
            lr=self.lr,
            alpha=0.9,
            eps=0.01,
        )
        optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip:
            nn.utils.clip_grad_norm_(
                self.q_net.parameters(),
                max_norm=self.config.clip_val,
            )
        optimizer.step()

        if self.t % self.config.target_update_freq == 0:
            self.update_target_net()
        if self.t % self.config.saving_freq == 0:
            self.save_model()

        self.logger.add_training_summary(
            loss,
            q,
            q_a,
            tq,
            tq_a,
            target,
            self.q_net.parameters(),
        )

        if self.t % self.config.log_image_freq == 0:
            self.logger.log_training_images(s, ns, a, r, self.t)

    def update_target_net(self):
        self.q_net.eval()
        self.target_q_net.eval()
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save_model(self):
        if not os.path.exists(self.config.model_path):
            os.makedirs(self.config.model_path)

        self.q_net.eval()
        torch.save(
            self.q_net.state_dict(),
            self.config.model_path + f"model-checkpoint-{self.t}.pt",
        )

class ConvNet(nn.Module):
    def __init__(self, output_units):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, output_units)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x