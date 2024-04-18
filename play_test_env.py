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
    input_size = (3, 5)
    config = {
        "replay_start_size": 5,
        "experience_buffer_size": 10,
        "mini_batch_size": 3,
        "gamma": 0.99,
        "learning_rate": 0.01,
        "anneal_steps": 100,
        "target_model_update_interval": 10,
    }
    
    env = EnvTest(input_size)
    player = LinearModelTestEnvPlayer(env, config)

    for i in range(2000):
        print(f"Playing step {i}...")
        player.step()