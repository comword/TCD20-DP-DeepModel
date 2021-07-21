import argparse
import json
from pathlib import Path
from glob import glob


class DatasetPrepare:

    def __init__(self, config: argparse.Namespace):
        self.config = config

    def run(self, input_dir: str):
        in_path = Path(input_dir)
        if not in_path.exists():
            raise ValueError("Input directory does not exist: " + input_dir)
        if not (in_path / "types.json"):
            raise ValueError(
                "Not a valid dataset folder, cannot find types.json in the input directory")

        with open(str(in_path / "types.json"), "r") as f:
            self.cls_types = json.load(f)

        for folder, cls_num in self.cls_types.items():
            if folder.startswith("Unused"):
                print("Skipping", folder)
                continue
            if not (in_path / folder).exists():
                raise ValueError(
                    "Not a valid dataset folder, cannot find category directory", folder)
            else:
                print(str(in_path / folder), "mapping to class", cls_num)

        files = glob(str(in_path / "**/*.mp4"), recursive=True)
        with open(str(in_path / "all.csv"), "w") as f:
            for video in files:
                v_split = str(video).split('/')
                label = self.cls_types[v_split[-2]]
                f.write("%s %d\n" % ("/".join(v_split[-2:]), label))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Video prepare')
    args.add_argument('-i', '--input', default=None, type=str, required=True,
                      help='input directory path')
    args_parsed = args.parse_args()
    prepare = DatasetPrepare(args_parsed)
    prepare.run(args_parsed.input)

# python src/utils/prepare2.py -i data/orig-front-videos
