from config import Config
from replay_buffer import ReplayBuffer
import datetime
import logging
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class RLPlayer(object):
    def __init__(self, env, config: Config, debug=False):
        self.env = env
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use a GPU device
        else:
            self.device = torch.device("cpu")  # Fallback to CPU if necessary
        self.init_model()

        self.debug = debug
        if self.debug:
            self.init_logger()

        self.replay_buffer = ReplayBuffer(
            n=self.config.replay_memory_size,
            state_depth=self.config.agent_history_length,
            device=self.device,
        )
        self.steps_played = 0
        self.reset()

        if self.debug:
            self.logger.debug(f"using config: {config.__dict__}")
            self.logger.debug(f"using device: {self.device}")
            self.logger.debug(f"prediction_model: {self.prediction_model}")
            self.logger.debug(f"target_model: {self.target_model}")
            self.replay_buffer.log(self.logger)

    def init_logger(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_name = f"logs/{now}-player_log"
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

    def init_model(self):
        self.prediction_model = ConvNet(self.env.action_space.n).to(self.device)
        self.target_model = ConvNet(self.env.action_space.n).to(self.device)
    
    def reset(self):
        observation, *_ = self.env.reset()
        self.replay_buffer.replace_last_frame(observation)

    def get_eps(self):
        if self.steps_played > self.config.final_exploration_frame:
            return self.config.final_exploration
        else:
            return self.config.initial_exploration + self.steps_played * self.config.eps_anneal_rate

    def get_action(self):
        if self.steps_played < self.config.replay_start_size:
            if self.debug:
                self.logger.debug(f"Taking a random action; only {self.steps_played} played")
            return self.env.action_space.sample(), None

        eps = self.get_eps()
        if random.random() < eps:
            if self.debug:
                self.logger.debug(f"Taking a random action due to exploration, eps = {eps}")

            return self.env.action_space.sample(), None
        else:
            with torch.no_grad():
                state = torch.unsqueeze(self.replay_buffer.get_last_state(), dim=0)
                action_values = self.prediction_model(state)
            action = torch.argmax(action_values, dim=1).item()
            action_value = action_values[0, action].item()

            if self.debug:
                self.logger.debug(f"Applied prediction model in action selection")
                self.logger.debug(f"State: {state}")
                self.logger.debug(f"State action values: {action_values}")
                self.logger.debug(f"Picked action: {action} with action value {action_value}")

            return action, action_value

    def step(self):
        if self.debug:
            self.logger.debug(f"=======================================================")
            self.logger.debug(f"    Kicking off step: {self.steps_played}")
            self.logger.debug(f"=======================================================")

        action, action_value = self.get_action()
        observation, reward, terminated, truncated, *_ = self.env.step(action)
        self.replay_buffer.add(
            action=action,
            reward=reward,
            done=(terminated or truncated),
            next_frame=observation,
        )
        if self.debug:
            self.logger.debug("Added a step to buffer")
            self.replay_buffer.log(self.logger)

        training_loss, lr = self.train()
        self.steps_played += 1

        return action, action_value, reward, terminated, training_loss, lr

    def get_lr(self):
        if self.steps_played >= self.config.lr_anneal_steps:
            return self.config.final_lr
        else:
            return self.config.initial_lr + self.steps_played * self.config.lr_anneal_rate

    def train(self):
        if self.steps_played < self.config.replay_start_size:
            if self.debug:
                self.logger.debug("Didn't train: not played enough steps")
            return None, None

        if self.steps_played % self.config.update_frequency != 0:
            if self.debug:
                self.logger.debug("Didn't train: not at the right step")
            return None, None

        lr = self.get_lr()
        optimizer = torch.optim.Adam(
            params=self.prediction_model.parameters(),
            lr=lr,
            # betas=(
            #     self.config.gradient_momentum,
            #     self.config.squared_gradient_momentum,
            # ),
            # eps=self.config.min_squared_gradient,
        )

        s, a, r, d, ns = self.replay_buffer.sample(self.config.mini_batch_size)
        if self.debug:
            self.logger.debug("Started model training")
            self.logger.debug(f"s.shape: {s.shape}")
            self.logger.debug(f"s: {s}")
            self.logger.debug(f"a: {a}")
            self.logger.debug(f"r: {r}")
            self.logger.debug(f"d: {d}")
            self.logger.debug(f"ns.shape: {ns.shape}")
            self.logger.debug(f"ns: {ns}")

        with torch.no_grad():
            q_next = self.target_model(ns)
            q_next_max, _ = torch.max(q_next, dim=1)
            q_next_max[d] = 0
            target = r + self.config.discount_factor * q_next_max
        q = self.prediction_model(s)
        q_a = q[torch.arange(q.size(0)), a]
        loss = nn.MSELoss()(q_a, target)

        prev_prediction_model_dict = self.prediction_model.state_dict()
        prev_target_model_dict = self.target_model.state_dict()

        optimizer.zero_grad()
        loss.backward()
        if self.config.grad_norm_clip:
            nn.utils.clip_grad_norm_(
                self.prediction_model.parameters(),
                max_norm=self.config.grad_norm_clip,
            )
        optimizer.step()

        if self.steps_played % self.config.target_netwrok_update_frequency == 0:
            self.target_model.load_state_dict(self.prediction_model.state_dict())

        if self.steps_played % self.config.model_saving_frequency == 0:
            torch.save(
                self.prediction_model.state_dict(),
                f"models/model-checkpoint-{self.steps_played}.pt",
            )

        if self.debug:
            self.logger.debug(f"q_next: {q_next}")
            self.logger.debug(f"q_next_max: {q_next_max}")
            self.logger.debug(f"target: {target}")
            self.logger.debug(f"q: {q}")
            self.logger.debug(f"q_a: {q_a}")
            self.logger.debug(f"loss: {loss}")
            self.logger.debug(f"prediction model before training step {prev_prediction_model_dict}")
            self.logger.debug(f"prediction model after training step {self.prediction_model.state_dict()}")
            self.logger.debug(f"target model before update: {prev_target_model_dict}")
            self.logger.debug(f"target model after update: {self.target_model.state_dict()}")
            self.logger.debug("Finished model training")

        return loss.item(), lr

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