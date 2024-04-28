from config import Config
from collections import deque
from replay_buffer import ReplayBuffer
import datetime
import gymnasium as gym
from gymnasium.experimental.wrappers import RecordVideoV0
from pong_wrapper import PongWrapper
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
import os

class RLPlayer(object):
    def __init__(self, env, config: Config, debug=False):
        self.env = env
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.init_models(config.model_path)

        self.debug = debug
        if self.debug:
            self.init_logger(config.log_path)

        self.init_log_summary()

        self.replay_buffer = ReplayBuffer(
            n=self.config.buffer_size,
            state_history=self.config.state_history,
            scale=self.config.obs_scale,
            device=self.device,
        )

        self.t = 0
        self.episode_reward = 0
        self.eps = self.config.eps_begin
        self.lr = self.config.lr_begin

        self.writer = SummaryWriter(log_dir=config.log_path + "/tb_logs/")

        if self.debug:
            self.logger.debug(f"config: {config.__dict__}")
            self.logger.debug(f"device: {self.device}")
            self.logger.debug(f"q_net: {self.q_net}")
            self.logger.debug(f"target_q_net: {self.target_q_net}")

    def init_logger(self, log_path):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_name = log_path + f"/{now}-player_log"
        self.logger = logging.getLogger("rl_player_logger")
        self.logger.setLevel(logging.DEBUG)
        
        debug_file_handler = logging.FileHandler(f"{log_file_name}.DEBUG")
        info_file_handler = logging.FileHandler(f"{log_file_name}.INFO")
        
        debug_file_handler.setLevel(logging.DEBUG)
        info_file_handler.setLevel(logging.INFO)

        format = logging.Formatter('[%(levelname)s] %(asctime)s: %(message)s')
        debug_file_handler.setFormatter(format)
        info_file_handler.setFormatter(format)

        self.logger.addHandler(debug_file_handler)
        self.logger.addHandler(info_file_handler)

    def init_models(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.q_net = ConvNet(self.env.action_space.n).to(self.device)
        self.target_q_net = ConvNet(self.env.action_space.n).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def init_log_summary(self):
        # action summary
        self.action_summary = deque(maxlen=self.config.log_window)
        self.q_summary = deque(maxlen=self.config.log_window)
        self.q_a_summary = deque(maxlen=self.config.log_window)

        # reward summary
        self.reward_summary = deque(maxlen=self.config.log_window)
        self.episode_reward_summary = deque(maxlen=self.config.log_window // 100)

        # training summary
        self.loss_summary = deque(maxlen=self.config.log_window)
        self.grad_norm_summary = deque(maxlen=self.config.log_window)
        self.training_q_summary = deque(maxlen=self.config.log_window)
        self.training_q_a_summary = deque(maxlen=self.config.log_window)
        self.q_next_max_summary = deque(maxlen=self.config.log_window)
        self.target_summary = deque(maxlen=self.config.log_window)
    
    def update_eps(self):
        self.eps += (self.config.eps_end - self.config.eps_begin) / self.config.eps_nsteps
        if self.t >= self.config.eps_nsteps:
            self.eps = self.config.eps_end
    
    def update_lr(self):
        self.lr += (self.config.lr_end - self.config.lr_begin) / self.config.lr_nsteps
        if self.t >= self.config.lr_nsteps:
            self.lr = self.config.lr_end

    def get_action(self, obs):
        self.t += 1

        if self.debug:
            self.logger.debug(f"=======================================================")
            self.logger.debug(f"    Kicking off step: {self.t}")
            self.logger.debug(f"=======================================================")
            self.logger.debug(f"Getting action for obs {obs}")

        self.replay_buffer.add_frame(obs)
        state = self.replay_buffer.get_last_state()

        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(torch.unsqueeze(state, dim=0))[0]

        if self.t < self.config.learning_start or random.random() < self.eps:
            action = self.env.action_space.sample()
            if self.debug:
                self.logger.debug(f"Taking a random action {action}")
        else:
            action = torch.argmax(q, dim=0).item()
            if self.debug:
                self.logger.debug(f"Applied Q-Net in action selection")
                self.logger.debug(f"State: {state}")
                self.logger.debug(f"Q values: {q}")
                self.logger.debug(f"Action: {action}")
                self.logger.debug(f"Action value: {q[action]}")

        if self.t % self.config.log_freq == 0:
            self.action_summary.append(action)
            self.q_a_summary.append(q[action].item())
            self.q_summary.extend(q.tolist())
        
        return state, action, q, self.eps

    def update(self, action, reward, done):
        self.replay_buffer.add_action(action)
        self.replay_buffer.add_reward(reward)
        self.replay_buffer.add_done(done)
        if self.debug:
            self.logger.debug("Added a step to buffer")
            self.replay_buffer.log(self.logger)

        loss, lr, training_info = self.train()

        self.episode_reward += reward
        if done:
            self.episode_reward_summary.append(self.episode_reward)
            self.episode_reward = 0
        if self.t % self.config.log_freq == 0:
            self.reward_summary.append(reward)

        if self.t % self.config.log_scalar_freq == 0:
            self.log_scalar_summary()
        if self.t % self.config.log_histogram_freq == 0:
            self.log_histogram_summary()

        if self.t % self.config.eval_freq == 0:
            self.evaluate()

        self.update_eps()
        self.update_lr()
        return loss, lr, training_info

    def train(self):
        if self.t < self.config.learning_start:
            if self.debug:
                self.logger.debug("Didn't train: not played enough steps")
            return None, None, None

        if self.t % self.config.learning_freq != 0:
            if self.debug:
                self.logger.debug("Didn't train: not at the right step")
            return None, None, None

        s, a, r, d, ns = self.replay_buffer.sample(self.config.batch_size)
        s = s[~d]
        a = a[~d]
        r = r[~d]
        ns = ns[~d]

        if len(s) == 0:
            if self.debug:
                self.logger.debug("Didn't train: empty sample")
            return None, None, None

        if self.debug:
            self.logger.debug("Started model training")
            self.logger.debug(f"s.shape: {s.shape}")
            self.logger.debug(f"s: {s}")
            self.logger.debug(f"a: {a}")
            self.logger.debug(f"r: {r}")
            self.logger.debug(f"d: {d}")
            self.logger.debug(f"ns.shape: {ns.shape}")
            self.logger.debug(f"ns: {ns}")

        self.target_q_net.eval()
        with torch.no_grad():
            q_next = self.target_q_net(ns)
            q_next_max, _ = torch.max(q_next, dim=1)
            target = r + self.config.gamma * q_next_max
        
        self.q_net.train()
        q = self.q_net(s)
        q_a = q[torch.arange(q.size(0)), a]
        loss = nn.HuberLoss()(q_a, target)

        optimizer = torch.optim.Adam(
            params=self.q_net.parameters(),
            lr=self.lr,
            betas=(0.9, 0.9),
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

        if self.t % self.config.log_freq == 0:
            self.loss_summary.append(loss.item())
            self.training_q_summary.extend(q.view(-1).tolist())
            self.training_q_a_summary.extend(q_a.tolist())
            self.q_next_max_summary.extend(q_next_max.tolist())
            self.target_summary.extend(target.tolist())

            grad_norm = 0
            for p in self.q_net.parameters():
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            self.grad_norm_summary.append(grad_norm)

        if self.t % self.config.target_update_freq == 0:
            self.q_net.eval()
            self.target_q_net.eval()
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        if self.t % self.config.saving_freq == 0:
            self.q_net.eval()
            torch.save(
                self.q_net.state_dict(),
                self.config.model_path + f"model-checkpoint-{self.t}.pt",
            )

        if self.debug:
            self.logger.debug(f"q_next: {q_next}")
            self.logger.debug(f"q_next_max: {q_next_max}")
            self.logger.debug(f"target: {target}")
            self.logger.debug(f"q: {q}")
            self.logger.debug(f"q_a: {q_a}")
            self.logger.debug(f"loss: {loss}")
            self.logger.debug(f"prediction model after training step {self.q_net.state_dict()}")
            self.logger.debug(f"target model after update: {self.target_q_net.state_dict()}")
            self.logger.debug("Finished model training")

        return loss.item(), self.lr, [s, a, r, ns, q_a, target]

    def evaluate(self):
        if self.debug:
            self.logger.debug(f"=======================================================")
            self.logger.debug(f"    Kicking off eval: {self.t}")
            self.logger.debug(f"=======================================================")

        env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
        env = PongWrapper(env, skip_frame=self.config.skip_frame)
        env = RecordVideoV0(env, self.config.record_path, name_prefix=f"eval-{self.t}", episode_trigger=lambda x: x==0)

        replay_buffer = ReplayBuffer(
            self.config.state_history,
            self.config.state_history,
            self.config.obs_scale,
            self.device,
        )

        action_summary = []
        q_summary = []
        q_a_summary = []
        reward_summary = []
        episode_reward_summary = []

        for i in range(self.config.num_episodes_test):
            episode_reward = 0
            obs, _ = env.reset()
            done = False
            while not done:
                replay_buffer.add_frame(obs)
                state = replay_buffer.get_last_state()

                self.q_net.eval()
                with torch.no_grad():
                    q = self.q_net(torch.unsqueeze(state, dim=0))[0]
                action = torch.argmax(q, dim=0).item()

                if self.debug:
                    self.logger.debug(f"Getting action for obs {obs}")
                    self.logger.debug(f"Applied Q-Net in action selection")
                    self.logger.debug(f"State: {state}")
                    self.logger.debug(f"Q values: {q}")
                    self.logger.debug(f"Action: {action}")
                    self.logger.debug(f"Action value: {q[action]}")

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                replay_buffer.add_action(action)
                replay_buffer.add_reward(reward)
                replay_buffer.add_done(done)
                if self.debug:
                    self.logger.debug("Added a step to buffer")
                    replay_buffer.log(self.logger)

                episode_reward += reward
                action_summary.append(action)
                reward_summary.append(reward)
                q_summary.extend(q.view(-1).tolist())
                q_a_summary.append(q[action].item())
            
            episode_reward_summary.append(episode_reward)
            if self.debug:
                self.logger.debug(f"Evaluated one episode. Reward: {episode_reward}")

        self.writer.add_scalar("0.scaler.eval.episode_reward.avg", mean(episode_reward_summary), self.t)
        self.writer.add_scalar("1.scaler.eval.episode_reward.min", min(episode_reward_summary), self.t)
        self.writer.add_scalar("1.scaler.eval.episode_reward.max", max(episode_reward_summary), self.t)
        self.writer.add_scalar("2.scalar.eval.reward.avg", mean(reward_summary), self.t)
        self.writer.add_scalar("2.scalar.eval.q.avg", mean(q_summary), self.t)
        self.writer.add_scalar("2.scalar.eval.q.min", min(q_summary), self.t)
        self.writer.add_scalar("2.scalar.eval.q.max", max(q_summary), self.t)
        self.writer.add_scalar("2.scalar.eval.q_action.avg", mean(q_a_summary), self.t)
        self.writer.add_scalar("2.scalar.eval.q_action.min", min(q_a_summary), self.t)
        self.writer.add_scalar("2.scalar.eval.q_action.max", max(q_a_summary), self.t)
        self.writer.add_histogram("3.histogram.eval.actions", torch.tensor(action_summary), self.t)
        self.writer.add_histogram("3.histogram.eval.rewards", torch.tensor(reward_summary), self.t)
        self.writer.add_histogram("3.histogram.eval.q", torch.tensor(q_summary), self.t)
        self.writer.add_histogram("3.histogram.eval.q_action", torch.tensor(q_a_summary), self.t)
        self.writer.flush()

    def log_scalar_summary(self):
        if len(self.episode_reward_summary) > 0:
            self.writer.add_scalar("0.scalar.episode.episode_reward.avg", mean(self.episode_reward_summary), self.t)
            self.writer.add_scalar("1.scalar.episode.episode_reward.min", min(self.episode_reward_summary), self.t)
            self.writer.add_scalar("1.scalar.episode.episode_reward.max", max(self.episode_reward_summary), self.t)
        self.writer.add_scalar("2.scalar.episode.reward.sum", sum(self.reward_summary), self.t)
        self.writer.add_scalar("2.scalar.episode.q.avg", mean(self.q_summary), self.t)
        self.writer.add_scalar("2.scalar.episode.q.min", min(self.q_summary), self.t)
        self.writer.add_scalar("2.scalar.episode.q.max", max(self.q_summary), self.t)
        self.writer.add_scalar("2.scalar.episode.q_action.avg", mean(self.q_a_summary), self.t)
        self.writer.add_scalar("2.scalar.episode.q_action.min", min(self.q_a_summary), self.t)
        self.writer.add_scalar("2.scalar.episode.q_action.max", max(self.q_a_summary), self.t)
        self.writer.add_scalar("2.scalar.episode.epsilon", self.eps, self.t)

        if len(self.loss_summary) > 0:
            self.writer.add_scalar("2.scalar.training.loss.avg", mean(self.loss_summary), self.t)
            self.writer.add_scalar("2.scalar.training.loss.min", min(self.loss_summary), self.t)
            self.writer.add_scalar("2.scalar.training.loss.max", max(self.loss_summary), self.t)
            self.writer.add_scalar("2.scalar.training.q.avg", mean(self.training_q_summary), self.t)
            self.writer.add_scalar("2.scalar.training.q.min", min(self.training_q_summary), self.t)
            self.writer.add_scalar("2.scalar.training.q.max", max(self.training_q_summary), self.t)
            self.writer.add_scalar("2.scalar.training.q_action.avg", mean(self.training_q_a_summary), self.t)
            self.writer.add_scalar("2.scalar.training.q_action.min", min(self.training_q_a_summary), self.t)
            self.writer.add_scalar("2.scalar.training.q_action.max", max(self.training_q_a_summary), self.t)
            self.writer.add_scalar("2.scalar.training.q_target.avg", mean(self.q_next_max_summary), self.t)
            self.writer.add_scalar("2.scalar.training.q_target.min", min(self.q_next_max_summary), self.t)
            self.writer.add_scalar("2.scalar.training.q_target.max", max(self.q_next_max_summary), self.t)
            self.writer.add_scalar("2.scalar.training.target.avg", mean(self.target_summary), self.t)
            self.writer.add_scalar("2.scalar.training.target.min", min(self.target_summary), self.t)
            self.writer.add_scalar("2.scalar.training.target.max", max(self.target_summary), self.t)
            self.writer.add_scalar("0.scalar.training.grad_norm.avg", mean(self.grad_norm_summary), self.t)
            self.writer.add_scalar("1.scalar.training.grad_norm.min", min(self.grad_norm_summary), self.t)
            self.writer.add_scalar("1.scalar.training.grad_norm.max", max(self.grad_norm_summary), self.t)
            self.writer.add_scalar("2.scalar.training.lr", self.lr, self.t)

        self.writer.flush()

    def log_histogram_summary(self):
        self.writer.add_histogram("3.histogram.episode.actions", torch.tensor(self.action_summary), self.t)
        self.writer.add_histogram("3.histogram.episode.rewards", torch.tensor(self.reward_summary), self.t)
        self.writer.add_histogram("3.histogram.episode.q", torch.tensor(self.q_summary), self.t)
        self.writer.add_histogram("3.histogram.episode.q_action", torch.tensor(self.q_a_summary), self.t)
        if len(self.training_q_a_summary) > 0:
            self.writer.add_histogram("3.histogram.training.q", torch.tensor(self.training_q_summary), self.t)
            self.writer.add_histogram("3.histogram.training.q_action", torch.tensor(self.training_q_a_summary), self.t)
            self.writer.add_histogram("3.histogram.training.q_target", torch.tensor(self.q_next_max_summary), self.t)
        self.log_model_summary("3.histogram.training.q_net", self.q_net)
        self.log_model_summary("3.histogram.training.target_q_net", self.target_q_net)
        self.writer.flush()

    def log_model_summary(self, model_name, model):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"{model_name}." + name, param, self.t)

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