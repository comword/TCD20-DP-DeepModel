import collections
import os
import sys
import argparse

src_dir = os.path.join("src")
sys.path.insert(0, src_dir)
from trainer import VideoTrainer
from utils import init
from parse_config import ConfigParser


def main(config: ConfigParser):
    config.config.update({
        "data_loader":{
            "args": {
                "validation_split": 0.0,
                "shuffle": False
            }
        }
    })
    data_loader, model = init(config)
    trainer = VideoTrainer(model, config, data_loader=data_loader)
    trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
