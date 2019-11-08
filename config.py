from easydict import EasyDict


config = EasyDict()

config.dataset_dir = './datasets/'
config.tensorboard_dir = './tensorboard/'
config.checkpoint_dir = './checkpoints/'

config.load_iter = 0
config.train_iters = 0
config.is_train = True
config.use_cuda = True
