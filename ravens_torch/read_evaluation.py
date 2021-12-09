# coding=utf-8
"""Script to read the results."""

import os
import pickle
from absl import app, flags

from ravens_torch.constants import EXPERIMENTS_DIR
from ravens_torch.utils.text import bold


flags.DEFINE_string('root_dir', EXPERIMENTS_DIR, help='Location of test data')
flags.DEFINE_string('task', 'block-insertion', help='Task to complete')
flags.DEFINE_string('agent', 'transporter',
                    help='Agent to perform Pick-and-Place')
flags.DEFINE_integer('n_demos', 100,
                     help='Number of training demos')
flags.DEFINE_integer('n_steps', 40000,
                     help='Number of training steps performed')
flags.DEFINE_bool('binary_reward', True,
                  help='If true, evaluation counts the number of successful episodes, else it sums the final rewards.')
flags.DEFINE_integer('n_runs', 1, '')
FLAGS = flags.FLAGS


def round_reward(reward):
    return 1 if reward > 0.999 else 0


def main(unused_argv):
    # For each training run.
    for train_run in range(FLAGS.n_runs):
        name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}'

        # Load results.
        with open(os.path.join(FLAGS.root_dir, f'{name}-{FLAGS.n_steps}.pkl'), 'rb') as f:
            data = pickle.load(f)

        rewards = [reward for reward, _ in data]
        if FLAGS.binary_reward:
            rewards = [round_reward(reward) for reward in rewards]
        success_rate = sum(rewards) / len(rewards)
        task_flags = [
            f"TASK: {FLAGS.task}",
            f"AGENTS: {FLAGS.agent}",
            f"NB. DEMOS: {FLAGS.n_demos}",
            f"NB. STEPS: {FLAGS.n_steps}",
            f"SUCCESS RATE: {100 * success_rate}%"
        ]
        print(bold("=" * 150 + "\n" + "\t\t".join(task_flags) + "\n" + "=" * 150))


if __name__ == '__main__':
    app.run(main)
