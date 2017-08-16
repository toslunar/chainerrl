from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import argparse
import os

import chainer

from chainerrl.agents.a3c import A3CModel
from chainerrl.agents import PPO
from chainerrl.envs import ale
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl import policy
from chainerrl import v_function

from dqn_phi import dqn_phi


class A3CFF(chainer.ChainList, A3CModel):

    def __init__(self, n_actions):
        self.head = links.NIPSDQNHead()
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        return self.pi(out), self.v(out)


def main():

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('rom', type=str)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--use-sdl', action='store_true')
    parser.add_argument('--max-episode-len', type=int, default=10000)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)

    # TODO(kataoka): anneal
    parser.add_argument('--lr', type=float, default=2.5e-4)

    parser.add_argument('--eval-interval', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')

    # TODO(kataoka): async
    parser.add_argument('--update-interval', type=int, default=128 * 8)

    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.set_defaults(use_sdl=False)
    args = parser.parse_args()

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    model = A3CFF(n_actions)
    opt = chainer.optimizers.Adam(args.lr)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))
    agent = PPO(model, opt,
                gpu=args.gpu,
                phi=dqn_phi,
                update_interval=args.update_interval,
                minibatch_size=args.batchsize, epochs=args.epochs,
                clip_eps=0.1,  # TODO(kataoka) anneal
                clip_eps_vf=None,
                )
    if args.load:
        agent.load(args.load)

    def make_env(test):
        env = ale.ALE(args.rom, use_sdl=args.use_sdl,
                      treat_life_lost_as_terminal=not test)
        if not test:
            misc.env_modifiers.make_reward_clipped(env, -1, 1)
        return env

    if args.demo:
        env = make_env(True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev: {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=make_env(False),
            eval_env=make_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=args.max_episode_len)


if __name__ == '__main__':
    main()