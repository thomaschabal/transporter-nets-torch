# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Script to plot training results."""

import os
import numpy as np
import pickle
from absl import app, flags

from ravens_torch.constants import EXPERIMENTS_DIR
from ravens_torch.utils import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', EXPERIMENTS_DIR, '')
flags.DEFINE_bool('disp', True, 'Whether to display the environment.')
flags.DEFINE_string('task', 'block-insertion', 'The task to run.')
flags.DEFINE_string('agent', 'transporter', 'The agent to run.')
flags.DEFINE_integer('n_demos', 100, 'Number of demos to run.')


def main(unused_argv):
    name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-'

    # Load and print results to console.
    path = FLAGS.root_dir
    curve = []

    for fname in sorted(os.listdir(path)):
        fname = os.path.join(path, fname)
        if name in fname and '.pkl' in fname:
            n_steps = int(fname[(fname.rfind('-') + 1):-4])
            data = pickle.load(open(fname, 'rb'))
            rewards = []
            for reward, _ in data:
                rewards.append(reward)
            score = np.mean(rewards)
            std = np.std(rewards)
            print(f'  {n_steps} steps:\t{score:.1f}%\t± {std:.1f}%')
            curve.append((n_steps, score, std))

    # Plot results over training steps.
    title = f'{FLAGS.agent} on {FLAGS.task} w/ {FLAGS.n_demos} demos'
    ylabel = 'Testing Task Success (%)'
    xlabel = '# of Training Steps'
    if FLAGS.disp:
        logs = {}
        curve = np.array(curve)
        logs[name] = (curve[:, 0], curve[:, 1], curve[:, 2])
        fname = os.path.join(path, f'{name}-plot.png')
        utils.plot(fname, title, ylabel, xlabel, data=logs, ylim=[0, 1])
        print(f'Done. Plot image saved to: {fname}')


if __name__ == '__main__':
    app.run(main)
