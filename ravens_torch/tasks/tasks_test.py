# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Integration tests for ravens_torch tasks."""

from absl.testing import absltest
from absl.testing import parameterized
from ravens_torch import tasks
from ravens_torch.environments import environment
from ravens_torch.constants import ENV_ASSETS_DIR


class TaskTest(parameterized.TestCase):

    def _create_env(self):
        assets_root = ENV_ASSETS_DIR
        env = environment.Environment(assets_root)
        env.seed(0)
        return env

    def _run_oracle_in_env(self, env):
        agent = env.task.oracle(env)
        obs = env.reset()
        info = None
        done = False
        for _ in range(10):
            act = agent.act(obs, info)
            obs, _, done, info = env.step(act)
            if done:
                break

    @parameterized.named_parameters((
        'AlignBoxCorner',
        tasks.AlignBoxCorner(),
    ), (
        'AssemblingKits',
        tasks.AssemblingKits(),
    ), (
        'AssemblingKitsEasy',
        tasks.AssemblingKitsEasy(),
    ), (
        'BlockInsertion',
        tasks.BlockInsertion(),
    ), (
        'ManipulatingRope',
        tasks.ManipulatingRope(),
    ), (
        'PackingBoxes',
        tasks.PackingBoxes(),
    ), (
        'PalletizingBoxes',
        tasks.PalletizingBoxes(),
    ), (
        'PlaceRedInGreen',
        tasks.PlaceRedInGreen(),
    ), (
        'StackBlockPyramid',
        tasks.StackBlockPyramid(),
    ), (
        'SweepingPiles',
        tasks.SweepingPiles(),
    ), (
        'TowersOfHanoi',
        tasks.TowersOfHanoi(),
    ))
    def test_all_tasks(self, ravens_task):
        env = self._create_env()
        env.set_task(ravens_task)
        self._run_oracle_in_env(env)


if __name__ == '__main__':
    absltest.main()
