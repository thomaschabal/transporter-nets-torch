import os
import datetime
import numpy as np
import torch

from ravens_torch.utils.text import bold


def get_log_dir(FLAGS):
    curr_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(FLAGS.train_dir, 'logs', FLAGS.agent, FLAGS.task,
                           curr_time, 'train')

    if FLAGS.verbose:
        print(f"Writing {bold('logs')} at {bold(log_dir)}")

    return log_dir


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_device(models, name, verbose=False):
    run_on_gpu = torch.cuda.is_available()
    if verbose:
        model_word = "models" if len(models) > 1 else "model "
        device_name = 'GPU' if run_on_gpu else 'CPU'
        print(f"Running {bold(name)} {model_word} on {bold(device_name)}")
    device = torch.device("cuda" if run_on_gpu else "cpu")

    for model in models:
        model.to(device)

    return device
