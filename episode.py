from config import Config
from collections import Counter
import matplotlib.pyplot as plt
import random
import os
from statistics import mean, stdev
import csv
from tqdm import tqdm

class Episode():
    def __init__(self, id, config: Config, n_actions):
        self.id = id
        self.t = 0
        self.actions = []
        self.q_list = []
        self.epsilons = []
        self.states = []
        self.next_obs_list = []
        self.rewards = []
        self.losses = []
        self.lrs = []
        self.training_info_list = []
        self.config = config
        self.done = False
        self.n_actions = n_actions

    def update_action(self, state, action, q, eps):
        self.states.append(state)
        self.actions.append(action)
        if q is not None:
            self.q_list.append(q)
        self.epsilons.append(eps)

    def update_feedback(self, next_obs, reward, done):
        self.next_obs_list.append(next_obs)
        self.rewards.append(reward)
        self.done = done
    
    def update_training(self, loss, lr, training_info):
        if loss is not None:
            self.losses.append(loss)
            self.lrs.append(lr)
            self.training_info_list.append(training_info)

    def log(self):
        return
        self.log_progress()
        if self.id % self.config.log_actions_freq == 0:
            self.log_actions()
        
        if self.id % self.config.log_training_freq == 0:
            self.log_training()

    def log_progress(self):
        q_values = [self.q_list[i][self.actions[i]].item() for i in range(len(self.q_list))]
        avg_q = 0 if len(q_values) == 0 else mean(q_values)
        stdev_q = 0 if len(q_values) == 0 else stdev(q_values)
        min_q = 0 if len(q_values) == 0 else min(q_values)
        max_q = 0 if len(q_values) == 0 else max(q_values)
        avg_eps = mean(self.epsilons)
        exp_rate = 1 - float(len(self.q_list)) / len(self.actions)
        total_reward = sum(self.rewards)

        avg_loss = 0 if len(self.losses) == 0 else mean(self.losses)
        avg_lr = 0 if len(self.lrs) == 0 else mean(self.lrs)

        action_distr = {k: v / len(self.actions) for k, v in Counter(self.actions).items()}
        action_distr_log = "["
        for i in range(self.n_actions):
            if i > 0:
                action_distr_log += ", "
            action_distr_log += f"{i}: {action_distr.get(i, 0):6.2%}"
        action_distr_log += "]"

        tqdm.write(f"Episode {self.id:6d} | "
                    f"t = {self.t:5d} | "
                    f"avg_q = {avg_q:8.5f} | "
                    f"stdev_q = {stdev_q:8.5f} | "
                    f"min_q = {min_q:8.5f} | "
                    f"max_q = {max_q:8.5f} | "
                    f"avg_eps = {avg_eps:5.3f} | "
                    f"exp_rate = {exp_rate:5.3f} | "
                    f"reward = {total_reward:2.0f} | "
                    f"avg_loss = {avg_loss:8.5f} | "
                    f"avg_lr = {avg_lr:8.5f} | "
                    f"action_distribution = {action_distr_log}")

        if not os.path.exists(self.config.log_path):
            os.makedirs(self.config.log_path)
        with open(self.config.log_path + f'/training_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.id,
                self.t,
                avg_q,
                stdev_q,
                avg_eps,
                total_reward,
                avg_loss,
                avg_lr,
            ])

    def log_actions(self):
        n_rows = min(self.config.n_actions_log, len(self.actions))
        start = random.randint(0, len(self.actions) - n_rows)
        fig, axes = plt.subplots(n_rows, 5, figsize=(84, 84))
        for i in range(n_rows):
            s = start + i
            state_images = self.states[s].numpy() * self.config.obs_scale
            assert len(state_images) == 4
            for j in range(4):
                axes[i, j].imshow(state_images[j], cmap='gray')
                axes[i, j].set_title(f"State {s}, obs {j}")
                axes[i, j].axis("off")

            axes[i, 4].imshow(self.next_obs_list[s], cmap='gray')
            axes[i, 4].set_title(f"Obs {s+1}, Action {self.actions[s]}")
            axes[i, 4].axis("off")

        plt.tight_layout()
        plt.savefig(
            self.config.log_path + f"action-plots-{self.id}.pdf",
            format='pdf',
            bbox_inches='tight',
        )
    
    def log_training(self):
        sample = random.sample(self.training_info_list, 1)
        s, a, r, ns, q_a, target = sample[0]
        n_rows = len(s)
        fig, axes = plt.subplots(n_rows, 8, figsize=(84, 84))
        for i in range(n_rows):
            s_images = s[i].numpy() * self.config.obs_scale
            ns_images = ns[i].numpy() * self.config.obs_scale
            for j in range(4):
                axes[i, j].imshow(s_images[j], cmap='gray')
                axes[i, j].axis("off")
            for j in range(4, 8):
                axes[i, j].imshow(ns_images[j-4], cmap='gray')
                axes[i, j].axis("off")
            
            axes[i, 0].set_title(f"action {a[i]}")
            axes[i, 1].set_title(f"reward {r[i]}")
            axes[i, 2].set_title(f"q_a {q_a[i]}")
            axes[i, 3].set_title(f"target {target[i]}")

        plt.tight_layout()
        plt.savefig(
            self.config.log_path + f"training-plots-{self.id}.pdf",
            format='pdf',
            bbox_inches='tight',
        )
