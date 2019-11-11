import os

import torch


def load_checkpoints(network, checkpoint_dir, name, load_iter):
    file = '{:05d}.ckpt'.format(load_iter)
    print("Loading {}...".format(file))
    network_path = os.path.join(checkpoint_dir, name, file)
    network.load_state_dict(torch.load(network_path, map_location=lambda storage, loc: storage))
