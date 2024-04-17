import logging
import random
import torch
import torch.nn as nn

class Player(object):
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
        self.logger = logging.getLogger("player_logger")
        self.logger.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        debug_file_handler = logging.FileHandler("player_log.DEBUG")
        info_file_handler = logging.FileHandler("player_log.INFO")
        
        console_handler.setLevel(logging.INFO)
        debug_file_handler.setLevel(logging.DEBUG)
        info_file_handler.setLevel(logging.INFO)

        format = logging.Formatter('%(asctime)s: [%(levelname)s] %(message)s')
        console_handler.setFormatter(format)
        debug_file_handler.setFormatter(format)
        info_file_handler.setFormatter(format)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(debug_file_handler)
        self.logger.addHandler(info_file_handler)
    
    def init_experience_buffer(self):
        self.state_buffer = torch.zeros(
            self.get_buffer_size(),
            *self.state.shape,
            dtype=torch.float,
        )
        self.next_state_buffer = torch.zeros(
            self.get_buffer_size(),
            *self.state.shape,
            dtype=torch.float,
        )
        self.end_state_buffer = torch.zeros(
            self.get_buffer_size(),
            dtype=torch.bool,
        )
        self.action_buffer = torch.zeros(
            self.get_buffer_size(),
            dtype=torch.long,
        )
        self.reward_buffer = torch.zeros(
            self.get_buffer_size(),
            dtype=torch.float,
        )

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

    def reset(self):
        self.num_steps_played_in_episode = 0
        self.episode_discounted_reward = 0

        observation = self.env.reset()
        self.state = None
        self.transition(observation)

    def get_action(self):
        eps = max(1 - self.total_steps_played * 0.9 / self.get_anneal_steps(), 0.1)

        self.logger.debug("Getting action")
        self.logger.debug(f"esp: {eps}")
        if (self.total_steps_played < self.config["random_start_steps"]
            or random.random() < eps):
            self.logger.debug("Returning a random action")
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                action_values = self.model(self.state)
            self.logger.debug(f"State action values: {action_values}")
            action = torch.argmax(action_values)
            self.logger.debug(f"Picked action: {action}")
            return action


    def step(self):
        self.logger.info(f"Playing a new step after {self.total_steps_played} steps")
        self.logger.info(f"Player state: {self.state}")

        action = self.get_action()
        self.logger.info(f"Taking action: {action}")

        observation, reward, terminated, _ = self.env.step(action)
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
        self.train()

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
                f"Discounted reward gathered in episode: \
                {self.episode_discounted_reward}"
            )
            self.reset()


    def update_experience_buffer(
            self, state, action, reward, next_state, terminated):

        def add_experience(buffer, experience):
            buffer[:-1] = buffer[1:].clone()
            self.logger.debug(f"buffer: {buffer}")
            self.logger.debug(f"experience: {experience}")
            buffer[-1] = experience
            self.logger.debug(f"new buffer: {buffer}")

        self.logger.debug("Updating experience buffer")
        add_experience(self.state_buffer, state)
        add_experience(self.action_buffer, action)
        add_experience(self.reward_buffer, reward)
        add_experience(self.next_state_buffer, next_state)
        add_experience(self.end_state_buffer, terminated)
        self.logger.debug("Experience buffer updated")

    def train(self):
        self.logger.debug("Training started")

        mini_batch_indexes = random.sample(
            range(self.get_buffer_size()),
            self.get_mini_batch_size()
        )

        self.logger.debug("Mini batch sampled: ")
        self.logger.debug(f"Indexes: {mini_batch-indexes}")

        s = self.state_buffer[mini_batch_indexes]
        self.logger.debug(f"States: {s}")

        a = self.action_buffer[mini_batch_indexes]
        self.logger.debug(f"Actions: {a}")

        r = self.reward_buffer[mini_batch_indexes]
        self.logger.debug(f"Rewards: {r}")

        ns = self.next_state_buffer[mini_batch_indexes]
        self.logger.debug(f"New states: {ns}")

        t = self.end_state_buffer[mini_batch_indexes]
        self.logger.debug(f"Termnated: {t}")

        optimizer = torch.optim.RMSprop(self.model.parameters(), self.get_learning_rate())

        with torch.no_grad():
            q_next = self.model(ns)
            self.logger.debug(f"q_next: {q_next}")

            q_next_max, _ = torch.max(q_next, dim=1)
            q_next_max[t] = 0
            self.logger.debug(f"q_next_max: {q_next_max}")

            target = r + self.get_gamma() * q_next_max
            self.logger.debug(f"target: {target}")

        q = self.model(s)
        self.logger.debug(f"q: {q}")
        q_a = q[torch.arange(q.size(0)), a]
        self.logger.debug(f"q_a: {q_a}")

        loss = nn.MSELoss()(q_a, target)
        self.logger.debug(f"loss: {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.logger.debug("Training finished")

    def init_model(self):
        raise

    def transition(self, observation):
        raise

class LinearModelPlayer(Player):
    def init_model(self):
        assert self.state is not None

        self.logger.debug(f"Iniitalizing linear model")
        self.logger.debug(f"in_features = {self.state.shape}")
        self.logger.debug(f"out_features = {self.env.action_space.n}")

        self.model = nn.Linear(
            in_features=self.state.shape[0],
            out_features=self.env.action_space.n,
            bias=True,
        )