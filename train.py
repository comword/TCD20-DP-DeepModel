import collections
import os
import sys
import argparse
from parse_config import ConfigParser

src_dir = os.path.join("src")
sys.path.insert(0, src_dir)
from trainer import VideoTrainer
from utils import init


def main(config):
    data_loader, model = init(config)
    if data_loader.hasSplitValidation():
        train_ds, val_ds = data_loader.getSplitDataset()

    trainer = VideoTrainer(model, config, train_ds=train_ds, val_ds=val_ds)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size'),
        CustomArgs(['--ec', '--epoch_count'],
                   type=int, target='trainer;epochs'),
    ]
    config = ConfigParser.from_args(parser, options)
    config.init_log()
    main(config)
