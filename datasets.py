import glob
import os

import numpy as np
import torch
from torch.utils import data


class MNIST(data.Dataset):
    def __init__(self, dataset_dir, split='train'):
        super(MNIST, self).__init__()

        self.dataset_dir = dataset_dir
        self.preprocess()

        self.data, self.labels = torch.load(os.path.join(self.dataset_dir, '{}.pt'.format(split)))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.dataset_dir, 'train.pt'))
                and os.path.exists(os.path.join(self.dataset_dir, 'val.pt'))
                and os.path.exists(os.path.join(self.dataset_dir, 'test.pt')))

    def preprocess(self):
        if self._check_exists():
            return

        raw_train_path = glob.glob('{}*train*'.format(self.dataset_dir))[0]
        raw_test_path = glob.glob('{}*test*'.format(self.dataset_dir))[0]
        raw_train_data = np.loadtxt(raw_train_path, dtype=np.float32)
        raw_test_data = np.loadtxt(raw_test_path, dtype=np.float32)

        train_set = (
            torch.as_tensor(raw_train_data[:10000, :-1]),
            torch.as_tensor(raw_train_data[:10000, -1], dtype=torch.long)
        )
        valid_set = (
            torch.as_tensor(raw_train_data[10000:, :-1]),
            torch.as_tensor(raw_train_data[10000:, -1], dtype=torch.long)
        )
        test_set = (
            torch.as_tensor(raw_test_data[:, :-1]),
            torch.as_tensor(raw_test_data[:, -1], dtype=torch.long)
        )

        torch.save(train_set, os.path.join(self.dataset_dir, 'train.pt'))
        torch.save(valid_set, os.path.join(self.dataset_dir, 'val.pt'))
        torch.save(test_set, os.path.join(self.dataset_dir, 'test.pt'))
        print("MNIST dataset is preprocessed.")
