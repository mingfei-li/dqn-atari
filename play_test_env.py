from players import LinearModelPlayer
from utils.test_env import EnvTest, ActionSpace
import math
import torch

class LinearModelTestEnvPlayer(LinearModelPlayer):
    def transition(self, observation):
        self.state = torch.tensor(
            observation.reshape(-1).astype(float),
            dtype=torch.float,
        )

if __name__ == "__main__":
    input_size = (5, 5, 1)
    config = {
        "random_start_steps": 50,
        "experience_buffer_size": 10,
        "mini_batch_size": 5,
        "gamma": 0.99,
        "learning_rate": 0.01,
        "anneal_steps": 1000
    }
    
    env = EnvTest(input_size)
    player = LinearModelTestEnvPlayer(env, config)

    for i in range(2000):
        print(f"Playing step {i}...")
        player.step()