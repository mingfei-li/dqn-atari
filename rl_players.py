from config import Config
from replay_buffer import ReplayBuffer
import datetime
import logging
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class RLPlayer(object):
    def __init__(self, env, config: Config, debug=False):
        self.env = env
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use a GPU device
        else:
            self.device = torch.device("cpu")  # Fallback to CPU if necessary
        self.init_models(config.model_path)

        self.debug = debug
        if self.debug:
            self.init_logger(config.log_path)

        self.replay_buffer = ReplayBuffer(
            n=self.config.buffer_size,
            state_history=self.config.state_history,
            device=self.device,
        )

        self.t = 0
        self.eps = self.config.eps_begin
        self.lr = self.config.lr_begin

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
    
    def reset(self, obs):
        self.replay_buffer.replace_last_frame(obs)

    def update_eps(self):
        self.eps += (self.config.eps_end - self.config.eps_begin) / self.config.eps_nsteps
        if self.t >= self.config.eps_nsteps:
            self.eps = self.config.eps_end
    
    def update_lr(self):
        self.lr += (self.config.lr_end - self.config.lr_begin) / self.config.lr_nsteps
        if self.t >= self.config.lr_nsteps:
            self.lr = self.config.lr_end

    def get_action(self):
        if self.t < self.config.learning_start or random.random() < self.eps:
            if self.debug:
                self.logger.debug(f"Taking a random action...")
            return self.env.action_space.sample(), None, self.eps

        self.q_net.eval()
        with torch.no_grad():
            state = torch.unsqueeze(self.replay_buffer.get_last_state(), dim=0)
            action_values = self.q_net(state)
        action = torch.argmax(action_values, dim=1).item()
        action_value = action_values[0, action].item()

        if self.debug:
            self.logger.debug(f"Applied prediction model in action selection")
            self.logger.debug(f"State: {state}")
            self.logger.debug(f"State action values: {action_values}")
            self.logger.debug(f"Picked action: {action} with action value {action_value}")

        return action, action_value, self.eps
    
    def update(self, action, obs, reward, done):
        self.t += 1
        if self.debug:
            self.logger.debug(f"=======================================================")
            self.logger.debug(f"    Kicking off update: {self.t}")
            self.logger.debug(f"=======================================================")

        self.replay_buffer.add(
            action=action,
            reward=reward,
            done=done,
            next_frame=obs,
        )
        if self.debug:
            self.logger.debug("Added a step to buffer")
            self.replay_buffer.log(self.logger)

        loss, lr = self.train()

        self.update_eps()
        self.update_lr()

        return loss, lr

    def train(self):
        if self.t < self.config.learning_start:
            if self.debug:
                self.logger.debug("Didn't train: not played enough steps")
            return None, None

        if self.t % self.config.learning_freq != 0:
            if self.debug:
                self.logger.debug("Didn't train: not at the right step")
            return None, None

        s, a, r, d, ns = self.replay_buffer.sample(self.config.batch_size)
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
            q_next_max[d] = 0
            target = r + self.config.gamma * q_next_max
        
        self.q_net.train()
        q = self.q_net(s)
        q_a = q[torch.arange(q.size(0)), a]
        loss = nn.MSELoss()(q_a, target)

        optimizer = torch.optim.Adam(
            params=self.q_net.parameters(),
            lr=self.lr,
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
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        if self.t % self.config.saving_freq == 0:
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

        return loss.item(), self.lr

class ConvNet(nn.Module):
    def __init__(self, output_units):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_units)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x