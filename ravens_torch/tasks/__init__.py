# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Ravens tasks."""

from ravens_torch.tasks.align_box_corner import AlignBoxCorner
from ravens_torch.tasks.assembling_kits import AssemblingKits
from ravens_torch.tasks.assembling_kits import AssemblingKitsEasy
from ravens_torch.tasks.block_insertion import BlockInsertion
from ravens_torch.tasks.block_insertion import BlockInsertionEasy
from ravens_torch.tasks.block_insertion import BlockInsertionNoFixture
from ravens_torch.tasks.block_insertion import BlockInsertionSixDof
from ravens_torch.tasks.block_insertion import BlockInsertionTranslation
from ravens_torch.tasks.manipulating_rope import ManipulatingRope
from ravens_torch.tasks.packing_boxes import PackingBoxes
from ravens_torch.tasks.palletizing_boxes import PalletizingBoxes
from ravens_torch.tasks.place_red_in_green import PlaceRedInGreen
from ravens_torch.tasks.stack_block_pyramid import StackBlockPyramid
from ravens_torch.tasks.sweeping_piles import SweepingPiles
from ravens_torch.tasks.task import Task
from ravens_torch.tasks.towers_of_hanoi import TowersOfHanoi

names = {
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,
    'palletizing-boxes': PalletizingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'stack-block-pyramid': StackBlockPyramid,
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi
}
