# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens
"""Data collection script."""

import os
import numpy as np
from absl import app, flags

from ravens_torch import tasks
from ravens_torch.constants import EXPERIMENTS_DIR, ENV_ASSETS_DIR
from ravens_torch.dataset import Dataset
from ravens_torch.environments.environment import Environment


flags.DEFINE_string('assets_root', ENV_ASSETS_DIR, '')
flags.DEFINE_string('data_dir', EXPERIMENTS_DIR, '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'block-insertion', '')
flags.DEFINE_string('mode', 'test', '')
flags.DEFINE_integer('n', 1000, '')

FLAGS = flags.FLAGS


def main(unused_argv):

    # Initialize environment and task.
    env = Environment(
        FLAGS.assets_root,
        disp=FLAGS.disp,
        shared_memory=FLAGS.shared_memory,
        hz=480)
    task = tasks.names[FLAGS.task]()
    task.mode = FLAGS.mode

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    dataset = Dataset(os.path.join(
        FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'))

    # Train seeds are even and test seeds are odd.
    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (task.mode == 'test') else -2

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < FLAGS.n:
        print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
        episode, total_reward = [], 0
        seed += 2
        np.random.seed(seed)
        env.set_task(task)
        obs = env.reset()
        info = None
        reward = 0
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            # print('Acting...', act)
            episode.append((obs, act, reward, info))
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward} Done: {done}')
            if done:
                break
        episode.append((obs, None, reward, info))

        # Only save completed demonstrations.
        if total_reward > 0.99:
            dataset.add(seed, episode)


if __name__ == '__main__':
    app.run(main)
