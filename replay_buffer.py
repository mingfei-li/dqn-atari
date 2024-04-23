from config import Config
import random
import torch
import torch.nn as nn
import logging
import numpy as np

class ReplayBuffer(object):
    def __init__(self, n, state_depth, device):
        self.n = n
        self.state_depth = state_depth
        self.device = device

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

    def replace_last_frame(self, frame):
        if self.frames == None:
            self.add_frame(frame)
        else:
            self.frames[self.back] = torch.tensor(frame, device=self.device)

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
        
        self.add_action(action)
        self.add_reward(reward)
        self.add_done(done)
        self.add_frame(next_frame)

    def sample(self, batch_size):
        # when sample is called, the last frame 
        # doesn't have action, reward and done added yet
        # so we shouldn't sample the last frame into "current states"
        # it should only be smapled into "next states"

        if self.is_full:
            indexes = [(i % self.n) for i in random.sample(
                range(self.back + self.state_depth, self.back + self.n),
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
    b = ReplayBuffer(n=8, state_depth=4, device=torch.device("cpu"))
    
    b.replace_last_frame(np.full(shape, 1))
    b.add(1, 1, False, np.full(shape, 2))

    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.full(shape, 1),
                           torch.full(shape, 2),
                       ],
                       dim=0)
                    )
    
    b.add(2, 2, False, np.full(shape, 3))
    b.add(3, 3, False, np.full(shape, 4))

    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.full(shape, 1),
                           torch.full(shape, 2),
                           torch.full(shape, 3),
                           torch.full(shape, 4),
                       ],
                       dim=0)
                    )

    b.add(4, 4, True, np.full(shape, 5))
    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.full(shape, 5),
                       ],
                       dim=0)
                    )
    
    b.replace_last_frame(np.full(shape, 6))
    b.add(6, 6, False, np.full(shape, 7))
    b.add(7, 7, False, np.full(shape, 8))
    b.add(8, 8, True, np.full(shape, 9))

    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.full(shape, 9),
                       ],
                       dim=0)
                    )

    b.replace_last_frame(np.full(shape, 10))
    b.add(10, 10, False, np.full(shape, 11))
    b.add(11, 11, False, np.full(shape, 12))
    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.full(shape, 10),
                           torch.full(shape, 11),
                           torch.full(shape, 12),
                       ],
                       dim=0)
                    )

    b.add(12, 12, False, np.full(shape, 13))
    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.full(shape, 10),
                           torch.full(shape, 11),
                           torch.full(shape, 12),
                           torch.full(shape, 13),
                       ],
                       dim=0)
                    )

    b.add(13, 13, False, np.full(shape, 14))
    b.add(14, 14, False, np.full(shape, 15))
    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.full(shape, 12),
                           torch.full(shape, 13),
                           torch.full(shape, 14),
                           torch.full(shape, 15),
                       ],
                       dim=0)
                    )

    b.add(15, 15, True, np.full(shape, 16))
    b.replace_last_frame(np.full(shape, 17))
    b.add(17, 17, True, np.full(shape, 18))
    assert torch.equal(b.get_last_state(), 
                       torch.stack([
                           torch.zeros(shape),
                           torch.zeros(shape),
                           torch.full(shape, 17),
                           torch.full(shape, 18),
                       ],
                       dim=0)
                    )

if __name__ == "__main__":
    test_last_state()