# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Tests for ravens_torch.environments.environment."""

from absl.testing import absltest

from ravens_torch import tasks
from ravens_torch.environments import environment
from ravens_torch.constants import ENV_ASSETS_DIR


class EnvironmentTest(absltest.TestCase):

    def test_environment_action(self):
        env = environment.Environment(ENV_ASSETS_DIR)
        task = tasks.BlockInsertion()
        env.set_task(task)
        env.seed(0)
        agent = task.oracle(env)
        obs = env.reset()
        info = None
        done = False
        for _ in range(10):
            act = agent.act(obs, info)
            self.assertTrue(env.action_space.contains(act))
            obs, _, done, info = env.step(act)
            if done:
                break


if __name__ == '__main__':
    absltest.main()
