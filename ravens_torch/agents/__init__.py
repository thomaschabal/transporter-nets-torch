# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Ravens agents package."""

from ravens_torch.agents.conv_mlp import PickPlaceConvMlpAgent
from ravens_torch.agents.dummy import DummyAgent
from ravens_torch.agents.form2fit import Form2FitAgent
from ravens_torch.agents.gt_state import GtState6DAgent
from ravens_torch.agents.gt_state import GtStateAgent
from ravens_torch.agents.gt_state_2_step import GtState2StepAgent
from ravens_torch.agents.gt_state_2_step import GtState3Step6DAgent
from ravens_torch.agents.transporter import GoalNaiveTransporterAgent
from ravens_torch.agents.transporter import GoalTransporterAgent
from ravens_torch.agents.transporter import NoTransportTransporterAgent
from ravens_torch.agents.transporter import OriginalTransporterAgent
from ravens_torch.agents.transporter import PerPixelLossTransporterAgent
from ravens_torch.agents.transporter_6dof import Transporter6dAgent

names = {'dummy': DummyAgent,
         'transporter': OriginalTransporterAgent,
         'transporter_6d': Transporter6dAgent,
         'no_transport': NoTransportTransporterAgent,
         'per_pixel_loss': PerPixelLossTransporterAgent,
         'conv_mlp': PickPlaceConvMlpAgent,
         'form2fit': Form2FitAgent,
         'gt_state': GtStateAgent,
         'gt_state_2_step': GtState2StepAgent,
         'gt_state_6d': GtState6DAgent,
         'gt_state_6d_3_step': GtState3Step6DAgent,
         'transporter-goal': GoalTransporterAgent,
         'transporter-goal-naive': GoalNaiveTransporterAgent}
