from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np
from statistics import mean
import torch
from torchvision.utils import make_grid

class TensorboardLogger():
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(log_dir=self.config.log_path)
        # action summary
        self.action_summary = deque(maxlen=self.config.log_window)
        self.obs_summary = deque(maxlen=20)
        self.q_summary = deque(maxlen=self.config.log_window)
        self.q_a_summary = deque(maxlen=self.config.log_window)

        # reward summary
        self.reward_summary = deque(maxlen=self.config.log_window)
        self.episode_reward_summary = deque(maxlen=self.config.log_window // 500)
        self.episode_length_summary = deque(maxlen=self.config.log_window // 500)
        self.episode_reward = 0
        self.episode_length = 0

        # training summary
        self.loss_summary = deque(maxlen=self.config.log_window)
        self.grad_norm_summary = deque(maxlen=self.config.log_window)
        self.training_q_summary = deque(maxlen=self.config.log_window)
        self.training_q_a_summary = deque(maxlen=self.config.log_window)
        self.training_tq_summary = deque(maxlen=self.config.log_window)
        self.training_tq_a_summary = deque(maxlen=self.config.log_window)
        self.target_summary = deque(maxlen=self.config.log_window)

    def add_action_summary(self, action, obs, q):
        self.action_summary.append(action)
        self.obs_summary.append(obs)
        self.q_a_summary.append(q[action].item())
        self.q_summary.extend(q.tolist())

    def add_reward_summary(self, reward, done):
        self.reward_summary.append(reward)
        self.episode_reward += reward
        self.episode_length += 1
        if done:
            self.episode_reward_summary.append(self.episode_reward)
            self.episode_length_summary.append(self.episode_length)
            self.episode_reward = 0
            self.episode_length = 0

    def add_training_summary(self, loss, q, q_a, tq, tq_a, target, params):
        self.loss_summary.append(loss.item())
        self.training_q_summary.extend(q.view(-1).tolist())
        self.training_q_a_summary.extend(q_a.tolist())
        self.training_tq_summary.extend(tq.view(-1).tolist())
        self.training_tq_a_summary.extend(tq_a.tolist())
        self.target_summary.extend(target.tolist())

        grad_norm = 0
        for p in params:
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        self.grad_norm_summary.append(grad_norm)

    def log_training_images(self, s, ns, a, r, t):
        for idx in range(s.shape[0]):
            # s_img = s[idx].numpy()
            # ns_img = ns[idx].numpy()
            # images = np.expand_dims(np.stack([s_img, ns_img], axis=0), axis=1)
            # images = (images * self.config.obs_scale).astype(np.uint8)
            
            images = torch.cat([s[idx], ns[idx]], dim=0).unsqueeze(1)
            torch.set_printoptions(threshold=8 * 80 * 80)
            print(images.shape)
            print(images)
            grid = make_grid(torch.tensor(images), self.config.state_history)
            self.writer.add_image(
                f"training.{t}.images.{idx}.action.{a[idx]}.reward.{r[idx]}",
                grid,
                t,
            )
        self.writer.flush()

    def log(self, t, eps, lr, q_net, tq_net):
        if t % self.config.log_scalar_freq == 0:
            self.log_scalar_summary(t, eps, lr)
        if t % self.config.log_histogram_freq == 0:
            self.log_histogram_summary(t, q_net, tq_net)
        if t % self.config.log_image_freq == 0:
            self.log_image_summary(t)

    def log_scalar_summary(self, t, eps, lr):
        if len(self.episode_reward_summary) > 0:
            self.writer.add_scalar("0.scalar.episode.episode_reward.avg", mean(self.episode_reward_summary), t)
            self.writer.add_scalar("1.scalar.episode.episode_reward.min", min(self.episode_reward_summary), t)
            self.writer.add_scalar("1.scalar.episode.episode_reward.max", max(self.episode_reward_summary), t)
            self.writer.add_scalar("0.scalar.episode.episode_length.avg", mean(self.episode_length_summary), t)
            self.writer.add_scalar("1.scalar.episode.episode_length.min", min(self.episode_length_summary), t)
            self.writer.add_scalar("1.scalar.episode.episode_length.max", max(self.episode_length_summary), t)
        self.writer.add_scalar("2.scalar.episode.reward.sum", sum(self.reward_summary), t)
        self.writer.add_scalar("2.scalar.episode.q.avg", mean(self.q_summary), t)
        self.writer.add_scalar("2.scalar.episode.q.min", min(self.q_summary), t)
        self.writer.add_scalar("2.scalar.episode.q.max", max(self.q_summary), t)
        self.writer.add_scalar("2.scalar.episode.q_action.avg", mean(self.q_a_summary), t)
        self.writer.add_scalar("2.scalar.episode.q_action.min", min(self.q_a_summary), t)
        self.writer.add_scalar("2.scalar.episode.q_action.max", max(self.q_a_summary), t)
        self.writer.add_scalar("2.scalar.episode.epsilon", eps, t)

        if len(self.loss_summary) > 0:
            self.writer.add_scalar("2.scalar.training.loss.avg", mean(self.loss_summary), t)
            self.writer.add_scalar("2.scalar.training.loss.min", min(self.loss_summary), t)
            self.writer.add_scalar("2.scalar.training.loss.max", max(self.loss_summary), t)
            self.writer.add_scalar("2.scalar.training.q.avg", mean(self.training_q_summary), t)
            self.writer.add_scalar("2.scalar.training.q.min", min(self.training_q_summary), t)
            self.writer.add_scalar("2.scalar.training.q.max", max(self.training_q_summary), t)
            self.writer.add_scalar("2.scalar.training.q_action.avg", mean(self.training_q_a_summary), t)
            self.writer.add_scalar("2.scalar.training.q_action.min", min(self.training_q_a_summary), t)
            self.writer.add_scalar("2.scalar.training.q_action.max", max(self.training_q_a_summary), t)
            self.writer.add_scalar("2.scalar.training.tq.avg", mean(self.training_tq_summary), t)
            self.writer.add_scalar("2.scalar.training.tq.min", min(self.training_tq_summary), t)
            self.writer.add_scalar("2.scalar.training.tq.max", max(self.training_tq_summary), t)
            self.writer.add_scalar("2.scalar.training.tq_action.avg", mean(self.training_tq_a_summary), t)
            self.writer.add_scalar("2.scalar.training.tq_action.min", min(self.training_tq_a_summary), t)
            self.writer.add_scalar("2.scalar.training.tq_action.max", max(self.training_tq_a_summary), t)

            self.writer.add_scalar("2.scalar.training.target.avg", mean(self.target_summary), t)
            self.writer.add_scalar("2.scalar.training.target.min", min(self.target_summary), t)
            self.writer.add_scalar("2.scalar.training.target.max", max(self.target_summary), t)
            self.writer.add_scalar("0.scalar.training.grad_norm.avg", mean(self.grad_norm_summary), t)
            self.writer.add_scalar("1.scalar.training.grad_norm.min", min(self.grad_norm_summary), t)
            self.writer.add_scalar("1.scalar.training.grad_norm.max", max(self.grad_norm_summary), t)
            self.writer.add_scalar("2.scalar.training.lr", lr, t)

        self.writer.flush()

    def log_histogram_summary(self, t, q_net, tq_net):
        self.writer.add_histogram("3.histogram.episode.actions", torch.tensor(self.action_summary), t)
        self.writer.add_histogram("3.histogram.episode.rewards", torch.tensor(self.reward_summary), t)
        self.writer.add_histogram("3.histogram.episode.q", torch.tensor(self.q_summary), t)
        self.writer.add_histogram("3.histogram.episode.q_action", torch.tensor(self.q_a_summary), t)
        if len(self.training_q_a_summary) > 0:
            self.writer.add_histogram("3.histogram.training.q", torch.tensor(self.training_q_summary), t)
            self.writer.add_histogram("3.histogram.training.q_action", torch.tensor(self.training_q_a_summary), t)
            self.writer.add_histogram("3.histogram.training.tq", torch.tensor(self.training_tq_summary), t)
            self.writer.add_histogram("3.histogram.training.tq_action", torch.tensor(self.training_tq_a_summary), t)
        self.log_model_summary("3.histogram.training.q_net", q_net, t)
        self.log_model_summary("3.histogram.training.target_q_net", tq_net, t)
        self.writer.flush()

    def log_model_summary(self, model_name, model, t):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"{model_name}." + name, param, t)
    
    def log_image_summary(self, t):
        images = np.expand_dims(np.stack(self.obs_summary, axis=0), axis=1)
        images = torch.tensor(images)
        actions = '_'.join(map(str, list(self.action_summary)[-8:]))
        rewards = '_'.join(map(str, list(self.reward_summary)[-8:]))
        grid = make_grid(images, self.config.state_history)
        self.writer.add_image(
            f"episode.{t}.images.actions.{actions}.rewards.{rewards}",
            grid,
            t,
        )
        self.writer.flush()
