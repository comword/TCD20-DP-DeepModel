import argparse
import json
import numpy as np
from pathlib import Path
from shutil import copy
import cv2


class AnonymiAlgo:

    def __init__(self, config: argparse.Namespace):
        self.config = config
        if not (config.__contains__("haar") and config.__contains__("lbf")):
            raise ValueError("Config args not sufficient" + str(vars(config)))
        self.detector = cv2.CascadeClassifier(config.haar)
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(config.lbf)

    def run(self, input_dir: str, output_dir: str):
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        if not in_path.exists():
            raise ValueError("Input directory does not exist: " + input_dir)
        if not (in_path / "types.json"):
            raise ValueError(
                "Not a valid dataset folder, cannot find types.json in the input directory")
        out_path.mkdir(parents=True, exist_ok=True)

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

        copy(in_path / "types.json", out_path / "types.json")

    def process_video(self, path: str):
        pass

    def process_frame(self, image: np.ndarray):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_equal = cv2.equalizeHist(img_gray)
        faces = self.detector.detectMultiScale(img_equal, scaleFactor=1.1, minNeighbors=3,
                                               flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_FIND_BIGGEST_OBJECT, minSize=(30, 30))
        for face in faces:
            (x, y, w, d) = face

        _, landmarks = self.landmark_detector.fit(img_equal, faces)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Video anonymisation')
    args.add_argument('-a', '--haar', default=None, type=str, required=True,
                      help='haar cascade model path')
    args.add_argument('-l', '--lbf', default=None, type=str, required=True,
                      help='FacemarkLBF model path')
    args.add_argument('-i', '--input', default=None, type=str, required=True,
                      help='input directory path')
    args.add_argument('-o', '--output', default=None, type=str, required=True,
                      help='path directory to output')
    args_parsed = args.parse_args()
    anonCls = AnonymiAlgo(args_parsed)
    anonCls.run(args_parsed.input, args_parsed.output)

# python src/util/anonymise.py -a data/face-model/haarcascade_frontalface_alt2.xml -l data/face-model/lbfmodel.yaml -i data/originalVideos -o data/anonymisedVideos
