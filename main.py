import os
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
from datasets import MNIST
from models import SdA


def train(models, writer, device):
    train_dataset = MNIST(config.dataset_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataset = MNIST(config.dataset_dir, split='val')
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    train_iter = iter(train_loader)
    for i in range(len(config.hidden_features)):
        for step in range(config.train_iters):
            try:
                data, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                data, _ = next(train_iter)
            data = data.to(device)

            x = data
            for k in range(i - 1):
                x = models.layers[k].encoder(x)
            idx = list(range(x.size(1)))
            random.shuffle(idx)
            x_noise = x.clone()
            x_noise[:, idx[:int(x.size(1) * config.w_v)]] = 0

            x_rec = models.layers[i](x_noise)
            loss = models.bce_criterion(x_rec, x)

            models.da_optimizers[i].zero_grad()
            loss.backward()
            models.da_optimizers[i].step()

            if (step + 1) % config.print_step == 0:
                print("[{}] step: {}/{}, loss: {:.4f}".format(i, step + 1, config.train_iters, loss.item()))

    train_iter = iter(train_loader)
    for step in range(config.train_iters):
        try:
            data, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, labels = next(train_iter)
        data = data.to(device)
        labels = labels.to(device)

        logits = models(data)
        loss = models.ce_criterion(logits, labels)

        models.sda_optimizer.zero_grad()
        loss.backward()
        models.sda_optimizer.step()

        if (step + 1) % config.print_step == 0:
            print("step: {}/{}, loss: {:.4f}".format(step + 1, config.train_iters, loss.item()))

        if (step + 1) % config.tensorboard_step == 0:
            writer.add_scalar('Loss/train', loss.item(), step + 1)
            with torch.no_grad():
                data, labels = next(iter(val_loader))
                data = data.to(device)
                labels = labels.to(device)

                logits = models(data)
                loss = models.ce_criterion(logits, labels)
                writer.add_scalar('Loss/val', loss.item(), step + 1)

    checkpoint_path = os.path.join(config.checkpoint_dir, config.name, '{}.ckpt'.format(config.train_iters))
    torch.save(models.layers.state_dict(), checkpoint_path)


def test(models, device):
    pass


def main():
    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir)
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    device = torch.device('cuda:0' if config.use_cuda else 'cpu')
    models = SdA(config).to(device)
    for layer in models.layers:
        layer.to(device)
    if config.load_iter != 0:
        pass

    if config.is_train:
        models.train()
        writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, config.name))
        train(models, writer, device)
    else:
        models.eval()
        test(models, device)


if __name__ == '__main__':
    main()
