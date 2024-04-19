import datetime
import logging
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class RLPlayer(object):
    def __init__(self, env, config):
        self.init_logger()
        self.env = env
        self.config = config
        self.total_steps_played = 0
        self.total_reward = 0
        self.num_episode_played = 0
        self.num_steps_played_in_episode = 0
        self.episode_discounted_reward = 0
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
        
        console_handler = logging.StreamHandler()
        debug_file_handler = logging.FileHandler(f"{log_file_name}.DEBUG")
        info_file_handler = logging.FileHandler(f"{log_file_name}.INFO")
        
        console_handler.setLevel(logging.INFO)
        debug_file_handler.setLevel(logging.DEBUG)
        info_file_handler.setLevel(logging.INFO)

        format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        console_handler.setFormatter(format)
        debug_file_handler.setFormatter(format)
        info_file_handler.setFormatter(format)

        # self.logger.addHandler(console_handler)
        self.logger.addHandler(debug_file_handler)
        self.logger.addHandler(info_file_handler)
    
    def init_experience_buffer(self):
        self.state_buffer = []
        self.next_state_buffer = []
        self.end_state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def get_buffer_size(self):
        return self.config["experience_buffer_size"]

    def get_mini_batch_size(self):
        return self.config["mini_batch_size"]

    def get_gamma(self):
        return self.config["gamma"]

    def get_learning_rate(self):
        return self.config["learning_rate"]
    
    def get_anneal_steps(self):
        return self.config["anneal_steps"]
    
    def get_learning_start_step(self):
        return self.config["replay_start_size"]

    def get_target_model_update_interval(self):
        return self.config["target_model_update_interval"]

    def reset(self):
        self.logger.info("Player resets")
        self.num_steps_played_in_episode = 0
        self.episode_discounted_reward = 0

        observation, *_ = self.env.reset()
        self.state = None
        self.transition(observation)

    def get_action(self):
        eps = max(1 - self.total_steps_played * 0.9 / self.get_anneal_steps(), 0.1)

        self.logger.debug("Getting action")
        self.logger.debug(f"esp: {eps}")
        if (self.total_steps_played < self.get_learning_start_step()
            or random.random() < eps):
            self.logger.debug("Returning a random action")
            return self.env.action_space.sample(), None
        else:
            self.logger.debug(f"Applying prediction model in action selection")
            with torch.no_grad():
                action_values = self.prediction_model(self.state)
            self.logger.debug(f"State action values: {action_values}")
            action = torch.argmax(action_values)
            self.logger.debug(f"Picked action: {action}")
            return action, action_values[action]

    def step(self):
        self.logger.info(f"Playing a new step after {self.total_steps_played} steps")
        self.logger.info(f"Player state: {self.state}")

        action, action_value = self.get_action()
        self.logger.info(f"Taking action: {action}")

        observation, reward, terminated, truncated, *_ = self.env.step(action)
        terminated |= truncated
        self.logger.info(f"Reward: {reward}")
        self.logger.info(f"Terminated: {terminated}")

        prev_state = self.state
        self.transition(observation)
        self.logger.info(f"New state: {self.state}")

        self.update_experience_buffer(
            prev_state,
            action,
            reward,
            self.state,
            terminated,
        )

        if self.total_steps_played >= self.get_learning_start_step():
            training_loss = self.train()
        else:
            training_loss = None
            self.logger.debug("Skipped learning for the step")

        self.total_steps_played += 1
        self.total_reward += reward
        self.episode_discounted_reward += reward * (
            self.get_gamma() ** self.num_steps_played_in_episode)
        self.num_steps_played_in_episode += 1
        self.logger.info(f"Finihsed playing step {self.total_steps_played}")

        if terminated:
            self.num_episode_played += 1
            self.logger.info(
                f"Episode {self.num_episode_played} concluded")
            self.logger.info(
                f"Steps played in episode: {self.num_steps_played_in_episode}")
            self.logger.info(
                f"Discounted reward gathered in episode: {self.episode_discounted_reward}"
            )
            self.reset()
        
        return action, action_value, reward, terminated, training_loss


    def update_experience_buffer(
            self,
            state,
            action,
            reward,
            next_state,
            terminated):
        self.logger.debug("Updating experience buffer")
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.next_state_buffer.append(next_state)
        self.end_state_buffer.append(terminated)

        if len(self.state_buffer) == self.get_buffer_size() * 2:
            self.logger.debug("Buffer full; clearing the first half")
            self.logger.debug(f"State buffer: {self.state_buffer}")
            self.logger.debug(f"Action buffer: {self.action_buffer}")
            self.logger.debug(f"Reward buffer: {self.reward_buffer}")
            self.logger.debug(f"Next state buffer: {self.next_state_buffer}")
            self.logger.debug(f"End state buffer: {self.end_state_buffer}")

            self.state_buffer = self.state_buffer[-self.get_buffer_size():].copy()
            self.action_buffer = self.action_buffer[-self.get_buffer_size():].copy()
            self.reward_buffer = self.reward_buffer[-self.get_buffer_size():].copy()
            self.next_state_buffer = self.next_state_buffer[-self.get_buffer_size():].copy()
            self.end_state_buffer = self.end_state_buffer[-self.get_buffer_size():].copy()

        self.logger.debug("Experience buffer updated")
        self.logger.debug(f"State buffer: {self.state_buffer}")
        self.logger.debug(f"Action buffer: {self.action_buffer}")
        self.logger.debug(f"Reward buffer: {self.reward_buffer}")
        self.logger.debug(f"Next state buffer: {self.next_state_buffer}")
        self.logger.debug(f"End state buffer: {self.end_state_buffer}")

    def sample_experiences(self):
        mini_batch_indexes = random.sample(
            range(
                max(len(self.state_buffer) - self.get_buffer_size(), 0),
                len(self.state_buffer),
            ),
            self.get_mini_batch_size()
        )

        self.logger.debug("Mini batch sampled: ")
        self.logger.debug(f"Indexes: {mini_batch_indexes}")

        s = torch.stack([self.state_buffer[i] for i in mini_batch_indexes], dim=0)
        a = torch.tensor([self.action_buffer[i] for i in mini_batch_indexes])
        r = torch.tensor([self.reward_buffer[i] for i in mini_batch_indexes])
        ns = torch.stack([self.next_state_buffer[i] for i in mini_batch_indexes], dim=0)
        t = torch.tensor([self.end_state_buffer[i] for i in mini_batch_indexes])

        self.logger.debug(f"States: {s}")
        self.logger.debug(f"Actions: {a}")
        self.logger.debug(f"Rewards: {r}")
        self.logger.debug(f"New states: {ns}")
        self.logger.debug(f"Termnated: {t}")

        return s, a, r, ns, t

    def train(self):
        self.logger.debug("Training started")

        optimizer = torch.optim.RMSprop(
            self.prediction_model.parameters(),
            self.get_learning_rate()
        )
        s, a, r, ns, t = self.sample_experiences()

        with torch.no_grad():
            q_next = self.target_model(ns)
            self.logger.debug(f"q_next: {q_next}")

            q_next_max, _ = torch.max(q_next, dim=1)
            # q_next_max[t] = 0
            self.logger.debug(f"q_next_max: {q_next_max}")

            target = r + self.get_gamma() * q_next_max
            self.logger.debug(f"target: {target}")

        q = self.prediction_model(s)
        self.logger.debug(f"q: {q}")
        q_a = q[torch.arange(q.size(0)), a]
        self.logger.debug(f"q_a: {q_a}")

        loss = nn.MSELoss()(q_a, target)
        self.logger.debug(f"loss: {loss}")

        self.logger.debug(f"prediction model before training step {self.prediction_model.state_dict()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.logger.debug(f"prediction model after training step {self.prediction_model.state_dict()}")

        if self.total_steps_played % self.get_target_model_update_interval() == 0:
            self.logger.debug(f"Updating target model")
            self.logger.debug(f"target model before update: {self.target_model.state_dict()}")

            self.target_model.load_state_dict(self.prediction_model.state_dict())
            self.logger.debug(f"target model after update: {self.target_model.state_dict()}")

        self.logger.debug("Training finished")
        return loss

    def init_model(self):
        raise

    def transition(self, observation):
        raise

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

class TestEnvPlayer(RLPlayer):
    def transition(self, observation):
        self.logger.debug(f"State transiiton from {self.state}")
        self.logger.debug(f"observation: {observation}")
        self.state = torch.tensor(
            observation.reshape(-1).astype(float),
            dtype=torch.float,
        )
        self.logger.debug(f"State transiitoned to {self.state}")

class PongPlayer(RLPlayer):
    def __init__(self, env, config):
        self.last_frame = torch.zeros(84, 84)
        super().__init__(env, config)

    def transition(self, observation):
        self.logger.debug(f"Transitioning from state: {self.state}")
        new_state = torch.zeros(4, 84, 84)
        if self.state is not None:
            new_state[:3] = self.state.view(4, 84, 84)[1:].clone()
        new_state[3] = self.transform_frame(observation)
        self.state = new_state.view(-1)
        self.logger.debug(f"Transitioned to new state: {self.state}")
        self.logger.debug(f"New state shape: {self.state.shape}")

    def transform_frame(self, observation):
        frame = torch.tensor(observation).permute(2, 0, 1)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((110, 84)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        frame = transform(frame).squeeze(0)[18:102, :]
        combined_frame = torch.max(self.last_frame, frame)
        self.last_frame = frame
        return combined_frame

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
        }
        super().__init__(env, config)

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
        }
        super().__init__(env, config)