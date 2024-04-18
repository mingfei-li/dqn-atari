from players import LinearModelPlayer, Player
import gymnasium as gym
import math
import torch
from PIL import Image
import torchvision.transforms as transforms

class PongPlayer(Player):
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

class LinearModelPongPlayer(LinearModelPlayer, PongPlayer):
    pass

if __name__ == "__main__":
    config = {
        "replay_start_size": 50,
        "experience_buffer_size": 1000,
        "mini_batch_size": 32,
        "gamma": 0.99,
        "learning_rate": 0.01,
        "anneal_steps": 1000,
        "target_model_update_interval": 100,
    }
    
    env = gym.make("ALE/Pong-v5", render_mode="human")
    player = LinearModelPongPlayer(env, config)

    for i in range(10000):
        print(f"Playing step {i}...")
        player.step()
    
    env.close()