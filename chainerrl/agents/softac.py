from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()

import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl.agent import Agent
from chainerrl.agent import AttributeSavingMixin
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
# from chainerrl.recurrent import Recurrent
# from chainerrl.recurrent import RecurrentChainMixin
# from chainerrl.recurrent import state_kept
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import ReplayUpdater


class SoftActorCriticModel(chainer.Chain):

    def __init__(self, policy, q_function, v_function):
        super().__init__()
        with self.init_scope():
            self.policy = policy
            self.q_function = q_function
            self.v_function = v_function


class SoftActorCritic(AttributeSavingMixin, Agent):
    """Soft Actor-Critic

    See https://arxiv.org/abs/1801.01290

    Args:
    """

    saved_attributes = (
        'model',
        'target_v_function',
        'actor_optimizer',
        'q_optimizer',
        'v_optimizer',
    )

    def __init__(self, model, actor_optimizer, q_optimizer, v_optimizer,
                 replay_buffer,
                 gamma, # explorer,
                 entropy_coef,
                 gpu=None, replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000,
                 phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 episodic_update=False,
                 episodic_update_len=None,
                 logger=getLogger(__name__)):

        self.model = model

        if gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)

        self.xp = self.model.xp
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        # self.explorer = explorer
        self.entropy_coef = entropy_coef
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.average_q_decay = average_q_decay
        self.average_loss_decay = average_loss_decay
        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.v_optimizer = v_optimizer
        if episodic_update:
            update_func = self.update_from_episodes
        else:
            update_func = self.update
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=update_func,
            batchsize=minibatch_size,
            episodic_update=episodic_update,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.target_v_function = copy.deepcopy(self.model['v_function'])
        self.average_q = 0
        self.average_actor_loss = 0.0
        self.average_q_loss = 0.0
        self.average_v_loss = 0.0

        # Aliases for convenience
        self.policy = self.model['policy']
        self.q_function = self.model['q_function']
        self.v_function = self.model['v_function']

        self.sync_target_network()

    def sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.v_function,
            dst=self.target_v_function,
            method=self.target_update_method,
            tau=self.soft_update_tau)

    # Update V-function
    def compute_v_loss(self, batch):
        batch_next_state = batch['next_state']
        batch_rewards = batch['reward']
        batch_terminal = batch['is_state_terminal']
        batch_state = batch['state']
        batch_actions = batch['action']
        batch_next_actions = batch['next_action']
        batchsize = len(batch_rewards)

        with chainer.no_backprop_mode():
            target_v = (
                F.reshape(self.q_function(batch_state, batch_actions), (-1,))
                - (self.entropy_coef
                   * self.policy(batch_state).log_prob(batch_actions)))

        predict_v = F.reshape(self.v_function(batch_state), (-1,))

        loss = 0.5 * F.mean_squared_error(predict_v, target_v)

        # Update stats
        self.average_v_loss += (
            (1 - self.average_loss_decay)
            * (float(loss.data) - self.average_v_loss)
        )

        return loss


    # Update Q-function
    def compute_q_loss(self, batch):
        """Compute loss for critic.

        Preconditions:
          target_q_function must have seen up to s_t and a_t.
          target_policy must have seen up to s_t.
          q_function must have seen up to s_{t-1}.
        Postconditions:
          target_q_function must have seen up to s_{t+1} and a_{t+1}.
          target_policy must have seen up to s_{t+1}.
          q_function must have seen up to s_t.
        """

        batch_next_state = batch['next_state']
        batch_rewards = batch['reward']
        batch_terminal = batch['is_state_terminal']
        batch_state = batch['state']
        batch_actions = batch['action']
        batch_next_actions = batch['next_action']
        batchsize = len(batch_rewards)

        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                next_v = F.reshape(
                    self.target_v_function(batch_next_state), (-1,))

            # # Target Q-function observes s_{t+1} and a_{t+1}
            # if isinstance(self.target_q_function, Recurrent):
            #     self.target_q_function.update_state(
            #         batch_next_state, batch_next_actions)

            target_q = batch_rewards + self.gamma * \
                (1.0 - batch_terminal) * next_v

        # Estimated Q-function observes s_t and a_t
        predict_q = F.reshape(
            self.q_function(batch_state, batch_actions), (-1,))

        loss = 0.5 * F.mean_squared_error(predict_q, target_q)

        # Update stats
        self.average_q_loss += (
            (1 - self.average_loss_decay)
            * (float(loss.data) - self.average_q_loss)
        )

        return loss

    def compute_actor_loss(self, batch):
        """Compute loss for actor.

        Preconditions:
          q_function must have seen up to s_{t-1} and s_{t-1}.
          policy must have seen up to s_{t-1}.
        Preconditions:
          q_function must have seen up to s_t and s_t.
          policy must have seen up to s_t.
        """

        batch_state = batch['state']
        batch_action = batch['action']

        log_prob = self.policy(batch_state).log_prob(batch_action)

        with chainer.no_backprop_mode():
            advantage = (
                F.reshape(self.q_function(batch_state, batch_action), (-1,))
                - self.entropy_coef * log_prob
                - F.reshape(self.v_function(batch_state), (-1,))
            )


        loss = -F.mean(log_prob * advantage)

        # Update stats
        self.average_actor_loss += (
            (1 - self.average_loss_decay)
            * (float(loss.data) - self.average_actor_loss)
        )

        return loss

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""

        batch = batch_experiences(experiences, self.xp, self.phi)
        self.v_optimizer.update(self.compute_v_loss, batch)
        self.q_optimizer.update(self.compute_q_loss, batch)
        self.actor_optimizer.update(self.compute_actor_loss, batch)

    def update_from_episodes(self, episodes, errors_out=None):
        assert False

        # Sort episodes desc by their lengths
        sorted_episodes = list(reversed(sorted(episodes, key=len)))
        max_epi_len = len(sorted_episodes[0])

        # Precompute all the input batches
        batches = []
        for i in range(max_epi_len):
            transitions = []
            for ep in sorted_episodes:
                if len(ep) <= i:
                    break
                transitions.append(ep[i])
            batch = batch_experiences(
                transitions, xp=self.xp, phi=self.phi)
            batches.append(batch)

        with self.model.state_reset():
            with self.target_model.state_reset():

                # Since the target model is evaluated one-step ahead,
                # its internal states need to be updated
                self.target_q_function.update_state(
                    batches[0]['state'], batches[0]['action'])
                self.target_policy(batches[0]['state'])

                # Update critic through time
                critic_loss = 0
                for batch in batches:
                    critic_loss += self.compute_critic_loss(batch)
                self.critic_optimizer.update(lambda: critic_loss / max_epi_len)

        with self.model.state_reset():

            # Update actor through time
            actor_loss = 0
            for batch in batches:
                actor_loss += self.compute_actor_loss(batch)
            self.actor_optimizer.update(lambda: actor_loss / max_epi_len)

    def act_and_train(self, state, reward):

        self.logger.debug('t:%s r:%s', self.t, reward)

        # greedy_action = self.act(state)
        # action = self.explorer.select_action(self.t, lambda: greedy_action)
        action = self._act(state, 'sample')
        self.t += 1

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

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

        self.replay_updater.update_if_necessary(self.t)

        return self.last_action

    def act(self, state):
        return self._act(state, 'argmax')

    def _act(self, state, sample_or_argmax):

        with chainer.using_config('train', False):
            s = batch_states([state], self.xp, self.phi)
            distrib = self.policy(s)
            if sample_or_argmax == 'sample':
                action = distrib.sample()
            elif sample_or_argmax == 'argmax':
                action = distrib.most_probable
            # Q is not needed here, but log it just for information
            q = self.q_function(s, action)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * float(q.data)

        self.logger.debug('t:%s a:%s q:%s',
                          self.t, action.data[0], q.data)
        return cuda.to_cpu(action.data[0])

    def stop_episode_and_train(self, state, reward, done=False):

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

        self.stop_episode()

    def stop_episode(self):
        self.last_state = None
        self.last_action = None
        # if isinstance(self.model, Recurrent):
        #     self.model.reset_state()
        self.replay_buffer.stop_current_episode()

    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_actor_loss', self.average_actor_loss),
            ('average_q_loss', self.average_q_loss),
            ('average_v_loss', self.average_v_loss),
        ]
