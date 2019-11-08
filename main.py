import os

import torch
from torch.utils.tensorboard import SummaryWriter

from config import config


def train(writer, device):
    pass


def test(device):
    pass


def main():
    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir)
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    device = torch.device('cuda:0' if config.use_cuda else 'cpu')
    if config.load_iter != 0:
        pass

    if config.is_train:
        writer = SummaryWriter(log_dir=config.tensorboard_dir)
        train(writer, device)
    else:
        test(device)


if __name__ == '__main__':
    main()
