from config import Config
import random
import torch
import torch.nn as nn
import logging
import numpy as np

class ReplayBuffer(object):
    # ideas borrowed from stanford cs234 course assignment starter code

    def __init__(self, n, state_history, scale, device):
        self.n = n
        self.state_history = state_history
        self.device = device
        self.scale = scale

        self.frames = None
        self.actions = None
        self.rewards = None
        self.done = None

    def initialize_buffer(self, shape):
        self.frames = torch.zeros(
            (self.n,) + shape,
            dtype=torch.uint8,
            device=self.device,
        )
        self.actions = torch.zeros(
            self.n, dtype=torch.int32, device=self.device)
        self.rewards = torch.zeros(
            self.n, dtype=torch.float16, device=self.device)
        self.done = torch.zeros(
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

        self.frames[self.back] = torch.tensor(frame, device=self.device)

    def add_action(self, action):
        self.actions[self.back] = action
    
    def add_reward(self, reward):
        self.rewards[self.back] = reward
    
    def add_done(self, done):
        self.done[self.back] = done

    def _get_state_at_index(self, index):
        need_wrapping_or_padding = index-self.state_history+1 < 0
        for i in range(1, self.state_history):
            if self.done[(index - i) % self.n]:
                need_wrapping_or_padding = True
                break

        if need_wrapping_or_padding:
            state = torch.zeros(
                (self.state_history,) + self.frames[index].shape,
                device=self.device,
            )
            for i in range(self.state_history):
                state[-(i+1)] = self.frames[index] / self.scale

                index -= 1
                if index < 0:
                    if self.is_full:
                        index = self.n - 1 
                    else:
                        break
                if self.done[index]:
                    break
            return state.float()
        else:
            start = index - self.state_history + 1
            end = index + 1
            return self.frames[start:end].float() / self.scale

    def get_last_state(self):
        return self._get_state_at_index(self.back)

    def sample(self, batch_size):
        # when sample is called, the last frame 
        # doesn't have action, reward and done added yet
        # so we shouldn't sample the last frame into "current states"
        # it should only be smapled into "next states"

        if self.is_full:
            indexes = [(i % self.n) for i in random.sample(
                range(self.back + self.state_history, self.back + self.n),
                batch_size,
            )]
        else:
            indexes = random.sample(
                range(self.back),
                batch_size,
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
        logger.debug(f"replay_buffer.is_full: {self.is_full}")
        logger.debug(f"replay_buffer.last_state: {self.get_last_state()}")


def test_last_state():
    shape = (84, 84)
    b = ReplayBuffer(n=8, state_history=4, scale=1, device=torch.device("cpu"))
    
    b.add_frame(np.full(shape, 1))
    b.add_action(1)
    b.add_reward(1)
    b.add_done(False)
    b.add_frame(np.full(shape, 2))

    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.full(shape, 1),
                           torch.full(shape, 2),
                       ],
                       dim=0)
                    )
    b.add_action(2)
    b.add_reward(2)
    b.add_done(False)
    b.add_frame(np.full(shape, 3))
    b.add_action(3)
    b.add_reward(3)
    b.add_done(False)
    b.add_frame(np.full(shape, 4))

    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.full(shape, 1),
                           torch.full(shape, 2),
                           torch.full(shape, 3),
                           torch.full(shape, 4),
                       ],
                       dim=0)
                    )

    b.add_action(4)
    b.add_reward(4)
    b.add_done(True)

    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.full(shape, 1),
                           torch.full(shape, 2),
                           torch.full(shape, 3),
                           torch.full(shape, 4),
                       ],
                       dim=0)
                    )

    b.add_frame(np.full(shape, 5))
    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.full(shape, 5),
                       ],
                       dim=0)
                    )
    b.add_action(5)
    b.add_reward(5)
    b.add_done(False)
    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.full(shape, 5),
                       ],
                       dim=0)
                    )

    b.add_frame(np.full(shape, 6))
    b.add_action(6)
    b.add_reward(6)
    b.add_done(False)

    b.add_frame(np.full(shape, 7))
    b.add_action(7)
    b.add_reward(7)
    b.add_done(True)

    b.add_frame(np.full(shape, 8))
    b.add_action(8)
    b.add_reward(8)
    b.add_done(False)
    
    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.full(shape, 8),
                       ],
                       dim=0)
                    )

    b.add_frame(np.full(shape, 9))
    b.add_action(9)
    b.add_reward(9)
    b.add_done(False)

    b.add_frame(np.full(shape, 10))
    b.add_action(10)
    b.add_reward(10)
    b.add_done(False)

    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.full(shape, 8),
                           torch.full(shape, 9),
                           torch.full(shape, 10),
                       ],
                       dim=0)
                    )

    b.add_frame(np.full(shape, 11))
    b.add_action(11)
    b.add_reward(11)
    b.add_done(False)

    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.full(shape, 8),
                           torch.full(shape, 9),
                           torch.full(shape, 10),
                           torch.full(shape, 11),
                       ],
                       dim=0)
                    )

    b.add_frame(np.full(shape, 12))
    b.add_action(12)
    b.add_reward(12)
    b.add_done(False)

    b.add_frame(np.full(shape, 13))
    b.add_action(13)
    b.add_reward(13)
    b.add_done(False)

    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.full(shape, 10),
                           torch.full(shape, 11),
                           torch.full(shape, 12),
                           torch.full(shape, 13),
                       ],
                       dim=0)
                    )

    b.add_frame(np.full(shape, 14))
    b.add_action(14)
    b.add_reward(14)
    b.add_done(True)

    b.add_frame(np.full(shape, 15))
    b.add_action(15)
    b.add_reward(15)
    b.add_done(False)

    b.add_frame(np.full(shape, 16))
    b.add_action(16)
    b.add_reward(16)
    b.add_done(False)

    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.full(shape, 15),
                           torch.full(shape, 16),
                       ],
                       dim=0)
                    )

if __name__ == "__main__":
    test_last_state()