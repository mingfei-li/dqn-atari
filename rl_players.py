import datetime
import logging
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class Config(object):
    def __init__(self, **kwargs):
        # Hyper-parameters: default parameters are from DeepMind Nature paper
        self.mini_batch_size = kwargs.get("mini_batch_size", 32)
        self.replay_memory_size = kwargs.get("replay_memory_size", 1_000_000)
        self.agent_history_length = kwargs.get("agent_history_length", 4)
        self.target_netwrok_update_frequency = kwargs.get("target_netwrok_update_frequency", 10_000)
        self.discount_factor = kwargs.get("discount_factor", 0.99)
        self.action_repeat = kwargs.get("action_repeat", 4)
        self.update_frequency = kwargs.get("update_frequency", 4)
        self.learning_rate = kwargs.get("learning_rate", 0.00025)
        self.gradient_momentum = kwargs.get("gradient_momentum", 0.95)
        self.squared_gradient_momentum = kwargs.get("squared_gradient_momentum", 0.95)
        self.min_squared_gradient = kwargs.get("min_squared_gradient", 0.01)
        self.initial_exploration = kwargs.get("initial_exploration", 1)
        self.final_exploration = kwargs.get("final_exploration", 0.1)
        self.final_exploration_frame = kwargs.get("final_exploration_frame", 1_000_000)
        self.replay_start_size = kwargs.get("replay_start_size", 50_000)
        self.no_op_max = kwargs.get("no_op_max", 30)
        self.anneal_rate = float(self.final_exploration - self.initial_exploration) / self.final_exploration_frame

class RLPlayer(object):
    def __init__(self, env, config: Config, debug_logging=False):
        self.debug_logging = debug_logging
        self.init_logger()

        self.env = env
        self.config = config
        self.logger.info(f"Using config: {config.__dict__}")

        self.steps_played = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use a GPU device
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")  # Fallback to CPU if necessary
            self.logger.info("GPU not available, using CPU instead.")
        self.reset()
        self.init_model()
        self.init_experience_buffer()
    
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

        format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        debug_file_handler.setFormatter(format)
        info_file_handler.setFormatter(format)

        self.logger.addHandler(debug_file_handler)
        self.logger.addHandler(info_file_handler)
    
    def init_experience_buffer(self):
        self.state_buffer = []
        self.next_state_buffer = []
        self.end_state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def reset(self):
        self.logger.info("Player resets")
        observation, *_ = self.env.reset()
        self.state = self.transition(None, observation)

    def get_action(self):
        if self.steps_played > self.config.final_exploration_frame:
            eps = self.config.final_exploration
        else:
            eps = self.config.initial_exploration + self.steps_played * self.config.anneal_rate

        self.logger.info("Getting action")
        if (self.steps_played < self.config.replay_start_size
            or random.random() < eps):
            self.logger.info("Returning a random action")
            return self.env.action_space.sample(), None
        else:
            self.logger.info(f"Applying prediction model in action selection")
            with torch.no_grad():
                input = torch.unsqueeze(self.state, dim=0)
                action_values = self.prediction_model(input)
            action = torch.argmax(action_values, dim=1).item()
            action_value = action_values[0, action].item()

            if self.debug_logging:
                self.logger.debug(f"State action values: {action_values}")
                self.logger.debug(f"Picked action: {action} with action value {action_value}")
            return action, action_value

    def step(self):
        self.logger.info(f"Playing a new step after {self.steps_played} steps")
        self.logger.info(f"Player state: {self.state}")

        action, action_value = self.get_action()
        self.logger.info(f"Taking action: {action}")

        observation, reward, terminated, truncated, *_ = self.env.step(action)
        done = terminated | truncated
        self.logger.info(f"Reward: {reward}")
        self.logger.info(f"Terminated: {terminated}")

        prev_state = self.state
        self.state = self.transition(prev_state, observation)
        self.logger.info(f"New state: {self.state}")

        self.update_experience_buffer(
            prev_state,
            action,
            reward,
            self.state,
            done,
        )

        if (self.steps_played >= self.config.replay_start_size
            and self.steps_played % self.config.update_frequency == 0):
            training_loss = self.train()
        else:
            training_loss = None
            self.logger.info("Skipped training for the step")

        self.steps_played += 1
        self.logger.info(f"Finihsed playing step {self.steps_played}")

        return action, action_value, reward, terminated, training_loss


    def update_experience_buffer(
            self,
            state,
            action,
            reward,
            next_state,
            terminated):
        self.logger.info("Updating experience buffer")
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.next_state_buffer.append(next_state)
        self.end_state_buffer.append(terminated)

        if len(self.state_buffer) == self.config.replay_memory_size * 2:
            self.logger.info("Buffer full; clearing the first half")
            if self.debug_logging:
                self.logger.debug(f"State buffer: {self.state_buffer}")
                self.logger.debug(f"Action buffer: {self.action_buffer}")
                self.logger.debug(f"Reward buffer: {self.reward_buffer}")
                self.logger.debug(f"Next state buffer: {self.next_state_buffer}")
                self.logger.debug(f"End state buffer: {self.end_state_buffer}")

            self.state_buffer = self.state_buffer[-self.config.replay_memory_size:].copy()
            self.action_buffer = self.action_buffer[-self.config.replay_memory_size:].copy()
            self.reward_buffer = self.reward_buffer[-self.config.replay_memory_size:].copy()
            self.next_state_buffer = self.next_state_buffer[-self.config.replay_memory_size:].copy()
            self.end_state_buffer = self.end_state_buffer[-self.config.replay_memory_size:].copy()

        self.logger.info("Experience buffer updated")

        if self.debug_logging:
            self.logger.debug(f"State buffer: {self.state_buffer}")
            self.logger.debug(f"Action buffer: {self.action_buffer}")
            self.logger.debug(f"Reward buffer: {self.reward_buffer}")
            self.logger.debug(f"Next state buffer: {self.next_state_buffer}")
            self.logger.debug(f"End state buffer: {self.end_state_buffer}")

    def sample_experiences(self):
        mini_batch_indexes = random.sample(
            range(
                max(len(self.state_buffer) - self.config.replay_memory_size, 0),
                len(self.state_buffer),
            ),
            self.config.mini_batch_size,
        )
        s = torch.stack([self.state_buffer[i] for i in mini_batch_indexes], dim=0)
        a = torch.tensor([self.action_buffer[i] for i in mini_batch_indexes], device=self.device)
        r = torch.tensor([self.reward_buffer[i] for i in mini_batch_indexes], device=self.device)
        ns = torch.stack([self.next_state_buffer[i] for i in mini_batch_indexes], dim=0)
        t = torch.tensor([self.end_state_buffer[i] for i in mini_batch_indexes], device=self.device)

        if self.debug_logging:
            self.logger.debug("Mini batch sampled: ")
            self.logger.debug(f"Indexes: {mini_batch_indexes}")
            self.logger.debug(f"States: {s}")
            self.logger.debug(f"Actions: {a}")
            self.logger.debug(f"Rewards: {r}")
            self.logger.debug(f"New states: {ns}")
            self.logger.debug(f"Termnated: {t}")

        return s, a, r, ns, t

    def train(self):
        self.logger.info("Training started")

        optimizer = torch.optim.Adam(
            params=self.prediction_model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.95, 0.95),
            eps=0.01,
        )

        s, a, r, ns, t = self.sample_experiences()

        with torch.no_grad():
            q_next = self.target_model(ns)
            self.logger.debug(f"q_next: {q_next}")

            q_next_max, _ = torch.max(q_next, dim=1)
            # q_next_max[t] = 0
            self.logger.debug(f"q_next_max: {q_next_max}")

            target = r + self.config.discount_factor * q_next_max
            self.logger.debug(f"target: {target}")

        q = self.prediction_model(s)
        q_a = q[torch.arange(q.size(0)), a]
        loss = nn.MSELoss()(q_a, target)

        if self.debug_logging:
            self.logger.debug(f"q: {q}")
            self.logger.debug(f"q_a: {q_a}")
            self.logger.debug(f"loss: {loss}")
            self.logger.debug(f"prediction model before training step {self.prediction_model.state_dict()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.debug_logging:
            self.logger.debug(f"prediction model after training step {self.prediction_model.state_dict()}")

        if self.steps_played % self.config.target_netwrok_update_frequency == 0:
            self.logger.info(f"Updating target model")

            if self.debug_logging:
                self.logger.debug(f"target model before update: {self.target_model.state_dict()}")

            self.target_model.load_state_dict(self.prediction_model.state_dict())

            if self.debug_logging:
                self.logger.debug(f"target model after update: {self.target_model.state_dict()}")

        self.logger.info("Training finished")
        return loss.item()

    def init_model(self):
        raise

    def transition(self, current_state, observation):
        raise

class TestEnvPlayer(RLPlayer):
    def transition(self, current_state, observation):
        self.logger.debug(f"State transiiton from {current_state}")
        self.logger.debug(f"observation: {observation}")
        new_state = torch.tensor(
            observation.astype(float),
            dtype=torch.float,
        )
        self.logger.debug(f"State transiitoned to {new_state}")
        return new_state

class PongPlayer(RLPlayer):
    def __init__(self, env, config):
        super().__init__(env, config)
        # self.last_frame = torch.zeros(84, 84, device=self.device)

    def transition(self, current_state, observation):
        self.logger.debug(f"Transitioning from state: {current_state}")
        self.logger.debug(f"Observation: {observation}")
        new_state = torch.zeros(4, 84, 84, device=self.device)
        if current_state is not None:
            new_state[:3] = current_state[1:].clone()
        new_state[3] = torch.tensor(observation, device=self.device)
        self.logger.debug(f"Transitioned to new state: {new_state}")
        self.logger.debug(f"New state shape: {new_state.shape}")
        return new_state

    # def transform_frame(self, observation):
    #     frame = torch.tensor(observation, device=self.device).permute(2, 0, 1)
    #     transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((110, 84)),
    #         transforms.Grayscale(),
    #         transforms.ToTensor(),
    #     ])
    #     frame = transform(frame).squeeze(0)[18:102, :]
    #     combined_frame = torch.max(self.last_frame, frame)
    #     self.last_frame = frame
    #     return combined_frame

class LinearModelPlayer(RLPlayer):
    def init_model(self):
        assert self.state is not None

        in_features = self.state.view(-1).shape[0]
        out_features=self.env.action_space.n

        self.logger.debug(f"Iniitalizing linear models")
        self.logger.debug(f"in_features = {in_features}")
        self.logger.debug(f"out_features = {out_features}")

        self.prediction_model = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
        )

        self.target_model = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
        )

class LinearModelTestEnvPlayer(LinearModelPlayer, TestEnvPlayer):
    def __init__(self, env):
        config = {
            "replay_start_size": 5,
            "experience_buffer_size": 10,
            "mini_batch_size": 3,
            "gamma": 0.99,
            "learning_rate": 0.01,
            "anneal_steps": 100,
            "target_model_update_interval": 10,
            "training_frequency": 4,
        }
        super().__init__(env, config)
    
    def transition(self, current_state, observation):
        return super().transition(current_state, observation).view(-1)

class LinearModelPongPlayer(LinearModelPlayer, PongPlayer):
    def __init__(self, env):
        config = {
            "replay_start_size": 50,
            "experience_buffer_size": 1000,
            "mini_batch_size": 32,
            "gamma": 0.99,
            "learning_rate": 0.01,
            "anneal_steps": 1000,
            "target_model_update_interval": 100,
            "training_frequency": 4,
        }
        super().__init__(env, config)
    
    def transition(self, current_state, observation):
        if current_state is not None:
            current_state = current_state.view(4, 84, 84)
        return super().transition(
            current_state,
            observation,
        ).view(-1)

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
    
class CNNModelPongPlayer(PongPlayer):
    def init_model(self):
        assert self.state is not None

        self.logger.info(f"Iniitalizing CNN models")
        self.prediction_model = ConvNet(self.env.action_space.n).to(self.device)
        self.target_model = ConvNet(self.env.action_space.n).to(self.device)
        if self.debug_logging:
            self.logger.debug(f"prediction_model: {self.prediction_model}")
            self.logger.debug(f"target_model: {self.target_model}")
        self.logger.info(f"Finished initializing CNN models")