from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 4 # 16 #
config.TRAIN.lr_init = 1e-4  #initial learning rate
config.TRAIN.beta1 = 0.9  # decay rate

## initialize G
config.TRAIN.n_epoch_init = 20#100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 30 #2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'data2017/DIV2K_valid_HR/'
config.VALID.video_test_path = 'video_test_4/'#'frames_gen/'#'frames_to_test/'
config.VALID.flow_test_path = 'flow_to_test_4/'#'flow_gen/'#'flow_to_test/'
config.VALID.lr_img_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
