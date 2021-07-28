from concurrent import futures
from pathlib import Path
import collections
import os
import sys
import json
import argparse
import time
import math
import logging
import grpc
import tensorflow as tf

src_dir = os.path.join("src")
sys.path.insert(0, src_dir)

from utils import init
from src import model
import src.gen_proto.student_pb2
import src.gen_proto.student_pb2_grpc as student_pb2__grpc
from src.services.StudentAppServicer import StudentAppServicer


def init_obj(cfg, name, module, *args, **kwargs):
    if "type" not in cfg[name] or "args" not in cfg[name]:
        return None
    module_name = cfg[name]['type']
    module_args = dict(cfg[name]['args'])
    assert all([k not in module_args for k in kwargs]
               ), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

def main(load_path: Path, address: str):
    if not (load_path.parent / "config.json").exists():
        print("Trained model config does not exist")
        return -1
    with open(str(load_path.parent / "config.json"), "r") as f:
        config = json.load(f)
    modelInst = init_obj(config, "arch", model)

    arch = tf.Variable("")
    ckpt = tf.train.Checkpoint(arch=arch,
                               model=modelInst)
    ckpt.read(str(load_path)).expect_partial()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    student_pb2__grpc.add_StudentAppServicer_to_server(StudentAppServicer(), server)
    server.add_insecure_port(address)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    args = argparse.ArgumentParser(description='TFLite Convertor')
    args.add_argument('-i', '--input', default=None, type=str, required=True,
                      help='input file path (default: None)')
    args.add_argument('-a', '--addr', default="[::]:8888", type=str, required=False,
                      help='listening address (default: [::]:8888)')
    args_parsed = args.parse_args()
    main(Path(args_parsed.input), args_parsed.addr)
