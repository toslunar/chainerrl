from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import gym
import numpy as np


class RandomizeAction(gym.ActionWrapper):
    """Apply a random action instead of the one sent by the agent.

    This wrapper can be used to make a stochastic env.

    For exploration during training, use explorers like
    chainerrl.explorers.ConstantEpsilonGreedy instead of this wrapper.

    Args:
        env (gym.Env): Env to wrap.
        random_fraction (float): Fraction of actions that will be replaced
            with a random action. It must be in [0, 1].
    """

    def __init__(self, env, random_fraction):
        super().__init__(env)
        assert 0 <= random_fraction <= 1
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.random_fraction = random_fraction
        self.original_action = None
        self.np_random = np.random.RandomState()

    def _action(self, action):
        self.original_action = original_action
        if self.np_random.rand() < self.random_fraction:
            return self.np_random.randint(self.env.action_space.n)
        else:
            return self.original_action

    def _seed(self, seed):
        self.np_random.seed(seed)
