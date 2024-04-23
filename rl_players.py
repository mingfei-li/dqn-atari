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

class ReplayBuffer(object):
    def __init__(self, config: Config, device):
        self.device = device
        self.n = config.replay_memory_size
        self.state_depth = config.agent_history_length

        self.frames = None
        self.actions = None
        self.rewards = None
        self.done = None

    def initialize_buffer(self, shape):
        self.frames = torch.empty(
            (self.n,) + shape,
            dtype=torch.uint8,
            device=self.device,
        )
        self.actions = torch.empty(
            self.n, dtype=torch.float16, device=self.device)
        self.rewards = torch.empty(
            self.n, dtype=torch.float16, device=self.device)
        self.done = torch.empty(
            self.n, dtype=torch.bool, device=self.device)

        self.back = -1
        self.is_full = False
    
    def add_frame(self, frame):
        if self.frames == None:
            self.initialize_buffer(frame.shape)

        self.back += 1
        if self.back == self.n:
            self.back = 0
            self.is_full = True

        self.frames[self.back] = frame

    def add_action(self, action):
        self.actions[self.back] = action
    
    def add_reward(self, reward):
        self.rewards[self.back] = reward
    
    def add_done(self, done):
        self.done[self.back] = done

    def _get_state_at_index(self, index):
        state = torch.zeros(
            (self.state_depth,) + self.frames[index].shape,
            device=self.device,
        )

        for i in range(self.state_depth):
            state[-(i+1)] = self.frames[index]

            index -= 1
            if index < 0:
                if self.is_full:
                    index = self.n - 1 
                else:
                    break
            if self.done[index]:
                break

        return state
    
    def get_last_state(self):
        return self._get_state_at_index(self.back)

    def add(self, action, reward, done, next_frame):
        # add actions, rewards, done before adding the next frame
        # so that the indexes line up
        
        self.replay_buffer.add_action(action)
        self.replay_buffer.add_reward(reward)
        self.replay_buffer.add_done(done)

        if not done:
            self.replay_buffer.add_frame(next_frame)

    def sample(self):
        # when sample is called, the last frame 
        # doesn't have action, reward and done added yet
        # so we shouldn't sample the last frame into "current states"
        # it should only be smapled into "next states"

        if self.is_full:
            indexes = [(i % self.n) for i in random.sample(
                range(self.back + self.state_depth, self.back + self.n),
                self.config.mini_batch_size,
            )]
        else:
            indexes = random.sample(
                range(self.back),
                self.config.mini_batch_size,
            )

        s = torch.stack([self._get_state_at_index(i) for i in indexes], dim=0)
        a = self.actions[indexes]
        r = self.rewards[indexes]
        d = self.done[indexes]
        ns = torch.stack(
            [self._get_state_at_index((i+1) % self.n) for i in indexes],
            dim=0,
        )

        return s, a, r, d, ns
    
    def log(self, logger):
        logger.debug(f"replay_buffer.frames: {self.frames}")
        logger.debug(f"replay_buffer.actions: {self.actions}")
        logger.debug(f"replay_buffer.rewards: {self.rewards}")
        logger.debug(f"replay_buffer.done: {self.done}")
        logger.debug(f"replay_buffer.back: {self.back}")


class RLPlayer(object):
    def __init__(self, env, config: Config, debug_logging=False):
        self.env = env
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use a GPU device
        else:
            self.device = torch.device("cpu")  # Fallback to CPU if necessary
        self.init_model()

        self.debug_logging = debug_logging
        if self.debug_logging:
            self.init_logger()

        self.replay_buffer = ReplayBuffer(config, self.device)
        self.steps_played = 0
        self.reset()

        if self.debug_logging:
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

        format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        debug_file_handler.setFormatter(format)
        info_file_handler.setFormatter(format)

        self.logger.addHandler(debug_file_handler)
        self.logger.addHandler(info_file_handler)
    
    def reset(self):
        observation, *_ = self.env.reset()
        self.replay_buffer.add_frame(observation)

    def get_action(self):
        if self.steps_played > self.config.final_exploration_frame:
            eps = self.config.final_exploration
        else:
            eps = self.config.initial_exploration + self.steps_played * self.config.anneal_rate

        if (self.steps_played < self.config.replay_start_size
            or random.random() < eps):

            if self.debug_logging:
                self.logger.debug("Returning a random action")

            return self.env.action_space.sample(), None
        else:
            with torch.no_grad():
                state = torch.unsqueeze(self.replay_buffer.get_last_state(), dim=0)
                action_values = self.prediction_model(state)
            action = torch.argmax(action_values, dim=1).item()
            action_value = action_values[0, action].item()

            if self.debug_logging:
                self.logger.debug(f"Applied prediction model in action selection")
                self.logger.debug(f"State: {action_values}")
                self.logger.debug(f"State action values: {action_values}")
                self.logger.debug(f"Picked action: {action} with action value {action_value}")

            return action, action_value

    def step(self):
        action, action_value = self.get_action()
        observation, reward, terminated, truncated, *_ = self.env.step(action)
        self.replay_buffer.add(
            action=action,
            reward=reward,
            done=(terminated or truncated),
            next_frame=observation,
        )
        if self.debug_logging:
            self.replay_buffer.log(self.logger)

        training_loss = self.train()
        self.steps_played += 1

        return action, action_value, reward, terminated, training_loss

    def train(self):
        if self.steps_played < self.config.replay_start_size:
            if self.debug_logging:
                self.logger.debug("Didn't train: not played enough steps")
            return None

        if self.steps_played % self.config.update_frequency != 0:
            if self.debug_logging:
                self.logger.debug("Didn't train: not at the right step")
            return None

        optimizer = torch.optim.Adam(
            params=self.prediction_model.parameters(),
            lr=self.config.learning_rate,
            betas=(
                self.config.gradient_momentum,
                self.config.squared_gradient_momentum,
            ),
            eps=self.config.min_squared_gradient,
        )

        s, a, r, d, ns = self.replay_buffer.sample()

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
        optimizer.step()
        if self.steps_played % self.config.target_netwrok_update_frequency == 0:
            self.target_model.load_state_dict(self.prediction_model.state_dict())


        if self.debug_logging:
            self.logger.debug("Finished model training")
            self.logger.debug(f"q_next: {q_next}")
            self.logger.debug(f"q_next_max: {q_next_max}")
            self.logger.debug(f"target: {target}")
            self.logger.debug(f"q: {q}")
            self.logger.debug(f"q_a: {q_a}")
            self.logger.debug(f"loss: {loss}")
            self.logger.debug(f"prediction model before training step {prev_prediction_model_dict()}")
            self.logger.debug(f"prediction model after training step {self.prediction_model.state_dict()}")
            self.logger.debug(f"target model before update: {prev_target_model_dict}")
            self.logger.debug(f"target model after update: {self.target_model.state_dict()}")

        return loss.item()

    def init_model(self):
        raise

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
    
class CNNModelPongPlayer(RLPlayer):
    def init_model(self):
        self.prediction_model = ConvNet(self.env.action_space.n).to(self.device)
        self.target_model = ConvNet(self.env.action_space.n).to(self.device)
