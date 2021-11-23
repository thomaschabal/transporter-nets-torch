# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens
"""Ravens main training script."""

from absl import app, flags
import numpy as np

from ravens_torch import agents
from ravens_torch.constants import EXPERIMENTS_DIR
from ravens_torch.dataset import load_data
from ravens_torch.utils import SummaryWriter
from ravens_torch.utils.initializers import get_log_dir, set_seed


flags.DEFINE_string('train_dir', EXPERIMENTS_DIR, '')
flags.DEFINE_string('data_dir', EXPERIMENTS_DIR, '')
flags.DEFINE_string('task', 'block-insertion', '')
flags.DEFINE_string('agent', 'transporter', '')
flags.DEFINE_float('hz', 240, '')
flags.DEFINE_integer('n_demos', 1, '')
flags.DEFINE_integer('n_steps', 40000, '')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_integer('interval', 100, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('gpu_limit', None, '')
flags.DEFINE_boolean('verbose', True, '')

FLAGS = flags.FLAGS


def main(unused_argv):
    # Load train and test datasets.
    train_dataset, test_dataset = load_data(FLAGS)

    # Run training from scratch multiple times.
    for train_run in range(FLAGS.n_runs):
        name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}'

        # Set up tensorboard logger.
        writer = SummaryWriter(get_log_dir(FLAGS))

        # Initialize agent.
        set_seed(train_run)
        agent = agents.names[FLAGS.agent](
            name, FLAGS.task, FLAGS.train_dir, verbose=FLAGS.verbose)

        # Limit random sampling during training to a fixed dataset.
        max_demos = train_dataset.n_episodes
        episodes = np.random.choice(range(max_demos), FLAGS.n_demos, False)
        train_dataset.set(episodes)

        # Train agent and save snapshots.
        while agent.total_steps < FLAGS.n_steps:
            for _ in range(FLAGS.interval):
                agent.train(train_dataset, writer)
            agent.validate(test_dataset, writer)
            if agent.total_steps % 1000 == 0:
                agent.save(FLAGS.verbose)


if __name__ == '__main__':
    app.run(main)
