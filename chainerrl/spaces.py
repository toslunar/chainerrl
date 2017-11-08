# The spaces in this module has a subset of the interface of gym.Space.
# This module does not import gym.

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()

from abc import ABCMeta
from abc import abstractmethod

import numpy as np


class Space(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def sample(self):
        raise NotImplementedError()


class Discrete(Space):
    def __init__(self, n, dtype=np.int32):
        dtype = np.dtype(dtype)
        n = dtype.type(n)
        self.n = n
        self.dtype = dtype

    def sample(self):
        return self.dtype.type(np.random.randint(self.n))


class Box(Space):
    def __init__(self, low, high, shape=None, dtype=np.float32):
        if shape is not None:
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
        else:
            assert low.shape == high.shape
            self.low = low.astype(dtype)
            self.high = high.astype(dtype)

    def sample(self):
        return np.random.uniform(self.low, self.high)


class Tuple(Space):
    def __init__(self, spaces):
        self.spaces = spaces

    def sample(self):
        return tuple(sp.sample() for sp in self.spaces)
