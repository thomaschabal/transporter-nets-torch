import torch
from tensorboardX import SummaryWriter as SumWriterX


class MeanMetrics:
    def __init__(self):
        self.init_values()

    def init_values(self):
        self.values = []

    def reset(self):
        self.init_values()

    def result(self):
        return torch.mean(self.values)

    def __call__(self, x):
        self.values.append(x)


class SummaryWriter(SumWriterX):
    def add_scalars(self, scalars):
        for (name, value, step) in scalars:
            self.add_scalar(name, value, step)
