import argparse
from pathlib import Path
import tensorflow as tf

def tflite_convert(load_path: Path, out_path: Path):
    # load_path to the SavedModel directory
    # out_path to the tflite model output file
    # Convert the model
    if not load_path.exists():
        print("Input path does not exist")
        return -1
    converter = tf.lite.TFLiteConverter.from_saved_model(load_path) 
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
