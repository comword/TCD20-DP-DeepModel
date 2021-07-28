import os, sys
import argparse
import json
from pathlib import Path
import tensorflow as tf
import numpy as np

src_dir = os.path.join("src")
sys.path.insert(0, src_dir)

from src import model


def init_obj(cfg, name, module, *args, **kwargs):
    if "type" not in cfg[name] or "args" not in cfg[name]:
        return None
    module_name = cfg[name]['type']
    module_args = dict(cfg[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

def tflite_convert(load_path: Path, out_path: Path):
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

    input_shape = config["arch"]["args"]["input_shape"]

    test = tf.random.uniform((1, input_shape[0], input_shape[1], input_shape[2], input_shape[3]))   # C, F, H, W
    frame_idx = np.arange(0, input_shape[1]).reshape((1, input_shape[1]))

    out = modelInst([test, frame_idx])
    print(out.shape)

    converter = tf.lite.TFLiteConverter.from_keras_model(modelInst)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    converter.experimental_new_converter = True
    converter.allow_custom_ops = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                        tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    # out_path e.g. out/posture.tflite
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    return 0


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='TFLite Convertor')
    args.add_argument('-i', '--input', default=None, type=str, required=True,
                      help='input file path (default: None)')
    args.add_argument('-o', '--output', default=None, type=str, required=True,
                      help='path to output (default: None)')
    args_parsed = args.parse_args()
    tflite_convert(Path(args_parsed.input), Path(args_parsed.output))
