import chainerrl


class NewDQN(chainerrl.agents.DQN):

    def input_initial_batch_to_target_model(self, batch, batch_sizes):
        super().input_initial_batch_to_target_model({
            'state': batch['state'][:batch_sizes[0]]
        })

    def update_from_episodes(self, episodes, errors_out=None):
        has_weights = isinstance(episodes, tuple)
        if has_weights:
            episodes, weights = episodes
            if errors_out is None:
                errors_out = []
        if errors_out is None:
            errors_out_step = None
        else:
            del errors_out[:]
            for _ in episodes:
                errors_out.append(0.0)
            errors_out_step = []

        batch_sizes = []
        transposed_episodes = []
        weights_step = []

        tmp = list(sorted(
            enumerate(episodes), key=lambda x: -len(x[1])))
        sorted_episodes = [elem[1] for elem in tmp]
        indices = [elem[0] for elem in tmp]  # argsort
        del tmp

        max_epi_len = len(sorted_episodes[0])
        for i in range(max_epi_len):
            transitions = []
            for ep, index in zip(sorted_episodes, indices):
                if len(ep) <= i:
                    break
                transitions.append(ep[i])
                if has_weights:
                    weights_step.append(weights[index])

            batch_sizes.append(len(transitions))
            transposed_episodes.extend(transitions)

        batch = batch_experiences(transposed_episodes,
                                  xp=self.xp,
                                  phi=self.phi,
                                  batch_states=self.batch_states)

        """
        if has_weights:
            sorted_weights = [weights[index] for index in indices]  # sorted by len(ep)
            batch['weights'] = self.xp.asarray(
                sorted_weights, dtype=self.xp.float32)
        """
        batch['weights'] = self.xp.asarray(
            weights_step, dtype=self.xp.float32)

        with state_reset(self.model):
            with state_reset(self.target_model):
                self.input_initial_batch_to_target_model({
                    'state': batch['state'][:batch_sizes[0]]
                })

                self._batch_sizes = self.xp.asarray(batch_sizes)
                loss = self._compute_loss(batch, self.gamma,
                                          errors_out=errors_out_step)

                # Update stats
                self.average_loss *= self.average_loss_decay
                self.average_loss += \
                    (1 - self.average_loss_decay) * float(loss.data)

                self.model.cleargrads()
                loss.backward()
                self.optimizer.update()

        if has_weights:
            sections = xp.cumsum(batch_sizes)
            assert sections[-1] == len(x)
            errors_out_step = chainer.functions.split_axis(errors_out_step, sections[:-1])
            for errors in errors_out_step:
                for err, index in zip(errors, indices):  # TODO: this is slow
                    errors_out[index] += err
            
            self.replay_buffer.update_errors(errors_out)

    def _compute_y_and_t(self, exp_batch, gamma):
        try:
            kwargs = {'batch_sizes': self._batch_sizes}
        except:
            kwargs = {}

        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        qout = self.model(batch_state, **kwargs)  #!!!

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch, gamma),
                (batch_size, 1))

        return batch_q, batch_q_target

    def _compute_target_values(self, exp_batch, gamma):
        try:
            kwargs = {'batch_sizes': self._batch_sizes}
        except:
            kwargs = {}

        batch_next_state = exp_batch['next_state']

        target_next_qout = self.target_model(batch_next_state, **kwargs)  #!!!
        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max

