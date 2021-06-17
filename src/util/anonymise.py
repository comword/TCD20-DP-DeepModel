import argparse
from pathlib import Path

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Video anonymisation')
    args.add_argument('-h', '--haar', default=None, type=str, required=True,
                      help='haar cascade model path')
    args.add_argument('-l', '--lbf', default=None, type=str, required=True,
                      help='FacemarkLBF model path')
    args.add_argument('-i', '--input', default=None, type=str, required=True,
                      help='input directory path')
    args.add_argument('-o', '--output', default=None, type=str, required=True,
                      help='path directory to output')
    args_parsed = args.parse_args()