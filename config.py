from easydict import EasyDict


config = EasyDict()

config.name = 'SdA-3_0.0'
config.dataset_dir = './datasets/mnist/'
config.tensorboard_dir = './tensorboard/'
config.checkpoint_dir = './checkpoints/'

config.batch_size = 64
config.num_workers = 4

config.input_features = 784
config.hidden_features = [2000, 320, 50]
config.classes = 10

config.lr = 1e-3
config.momentum = 0.9
config.weight_decay = 0
config.w_v = 0

config.print_step = 10
config.tensorboard_step = 100
config.load_iter = 0
config.train_iters = 5000
config.is_train = True
config.use_cuda = True
