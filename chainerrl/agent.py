from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import os

from chainer import serializers
from future.utils import with_metaclass
import numpy
import warnings

from chainerrl.misc.makedirs import makedirs


def load_npz_no_strict(filename, obj):
    try:
        serializers.load_npz(filename, obj)
    except KeyError as e:
        warnings.warn(repr(e))
        with numpy.load(filename) as f:
            d = serializers.NpzDeserializer(f, strict=False)
            d.load(obj)


class Agent(with_metaclass(ABCMeta, object)):
    """Abstract agent class."""

    @abstractmethod
    def act_and_train(self, obs, reward):
        """Select an action for training.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def act(self, obs):
        """Select an action for evaluation.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_episode_and_train(self, state, reward, done=False):
        """Observe consequences and prepare for a new episode.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_episode(self):
        """Prepare for a new episode.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, dirname):
        """Save internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, dirname):
        """Load internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_statistics(self):
        """Get statistics of the agent.

        Returns:
            List of two-item tuples. The first item in a tuple is a str that
            represents the name of item, while the second item is a value to be
            recorded.

            Example: [('average_loss': 0), ('average_value': 1), ...]
        """
        pass


class ReplayBufferingMixin(object):
    """Mixin that provides transition memorization."""

    def init_replay_buffering(self, replay_buffer):
        self.replay_buffer = replay_buffer
        self.last_state = None
        self.last_action = None

    @abstractmethod
    def exploring_act(self, state):
        """Select an action for training.

        Returns:
            ~object: action
        """

    @abstractmethod
    def train(self):
        """Train

        """

    def act_and_train(self, state, reward):
        action = self.exploring_act(state)
        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=state,
                next_action=action,
                is_state_terminal=False)

        self.last_state = state
        self.last_action = action

        self.train()
        return action

    def stop_episode_and_train(self, state, reward, done=False):
        """Observe a terminal state and a reward.

        This function must be called once when an episode terminates.
        """

        assert self.last_state is not None
        assert self.last_action is not None

        # Add a transition to the replay buffer
        self.replay_buffer.append(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=state,
            next_action=self.last_action,
            is_state_terminal=done)

        self.last_state = None
        self.last_action = None
        self.replay_buffer.stop_current_episode()

        self.stop_episode()


class AttributeSavingMixin(object):
    """Mixin that provides save and load functionalities."""

    @abstractproperty
    def saved_attributes(self):
        """Specify attribute names to save or load as a tuple of str."""
        pass

    def save(self, dirname):
        """Save internal states."""
        makedirs(dirname, exist_ok=True)
        for attr in self.saved_attributes:
            serializers.save_npz(
                os.path.join(dirname, '{}.npz'.format(attr)),
                getattr(self, attr))

    def load(self, dirname):
        """Load internal states."""
        for attr in self.saved_attributes:
            """Fix Chainer Issue #2772

            In Chainer v2, a (stateful) optimizer cannot be loaded from
            an npz saved before the first update.
            """
            load_npz_no_strict(
                os.path.join(dirname, '{}.npz'.format(attr)),
                getattr(self, attr))


class AsyncAgent(with_metaclass(ABCMeta, Agent)):
    """Abstract asynchronous agent class."""

    @abstractproperty
    def process_idx(self):
        """Index of process as integer, 0 for the representative process."""
        pass

    @abstractproperty
    def shared_attributes(self):
        """Tuple of names of shared attributes."""
        pass
