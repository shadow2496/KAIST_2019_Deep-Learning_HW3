from easydict import EasyDict


config = EasyDict()

config.dataset_dir = './datasets/'
config.tensorboard_dir = './tensorboard/'
config.checkpoint_dir = './checkpoints/'

config.input_features = 784
config.hidden_features = [2000, 1000, 500]
config.classes = 10

config.lr = 1e-3
config.momentum = 0.9
config.weight_decay = 0

config.load_iter = 0
config.train_iters = 0
config.is_train = True
config.use_cuda = True
