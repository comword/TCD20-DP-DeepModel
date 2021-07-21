import collections
import os
import sys
import argparse

src_dir = os.path.join("src")
sys.path.insert(0, src_dir)

from parse_config import ConfigParser
from utils import init
from trainer import VideoTrainer


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def main(config: ConfigParser):
    update(config.config, {
        "data_loader": {
            "args": {
                "validation_split": 0.5,
                "shuffle": True,
                "batch_size": 1
            }
        }
    })
    data_loader, model = init(config)
    trainer = VideoTrainer(model, config, data_loader=data_loader)
    results, logit, gTruth = trainer.test()


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
