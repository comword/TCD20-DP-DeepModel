import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from tqdm import tqdm
import cv2


class DatasetPrepare:

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    def __init__(self, config: argparse.Namespace):
        self.config = config
        if not (config.__contains__("haar") and config.__contains__("lbf")):
            raise ValueError("Config args not sufficient" + str(vars(config)))
        self.mask = config.mask
        if self.mask:
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

        # data files are 0fd582cc7c2bae69b44a233193caa577_0.jpg
        # csv is formatted as 0fd582cc7c2bae69b44a233193caa577 25 5
        self.table = pd.DataFrame(
            columns=["name", "fps", "label"])

        files = glob(str(in_path / "**/*.mp4"), recursive=True)
        for video in files:
            self.process_video(Path(video), out_path)

        self.table.to_csv(out_path / "videos.csv", index=False)

    def process_video(self, path: Path, out: Path):
        print("Processing video", path)
        label_name = str(path.parents[0]).split('/')[-1]
        if(not label_name in self.cls_types):
            print("Label for file", path,
                  "does not exist in types.json, skipping")
            return

        label = self.cls_types[label_name]
        cap = cv2.VideoCapture(str(path))

        if (cap.isOpened() == False):
            print("Error opening video file, skipping")
            cap.release()
            return
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        with tqdm(total=video_length) as pbar:
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                else:
                    filename = out / ("%s_%d.jpg" % (path.stem, frame_count))
                    ret = self.process_frame(frame, str(filename))
                    if ret == False:
                        print("Error processing video frame, skipping")
                        break
                    pbar.update(1)
                frame_count += 1

        self.table = self.table.append({
            "name": path.stem,
            "fps": round(cap.get(cv2.CAP_PROP_FPS)),
            "label": label,
        }, ignore_index=True)
        cap.release()

    def process_frame(self, image: np.ndarray, out: str):
        if self.mask:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_equal = cv2.equalizeHist(img_gray)
            faces = self.detector.detectMultiScale(img_equal, scaleFactor=1.1, minNeighbors=3,
                                                   flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_FIND_BIGGEST_OBJECT, minSize=(30, 30))
            for face in faces:
                (x, y, w, h) = face
                aspect_ratio = w / h
                if(0.75 < aspect_ratio and aspect_ratio < 1.3):
                    cx = round((x + w * 0.5))
                    cy = round((y + h * 0.5))
                    radius = round((w + h) * 0.27)
                    cv2.circle(image, (cx, cy), radius, (0, 0, 0), -1)
                else:
                    cv2.rectangle(image, (round(x), round(y)),
                                  (round((x + w - 1)), round((y + h - 1))),
                                  (0, 0, 0), -1)

            _, landmarks = self.landmark_detector.fit(img_equal, faces)
            for (idx, it) in enumerate(landmarks):
                color = self.colors[idx%3]
                for point in it[0]:
                    cx = round( point[0] )
                    cy = round( point[1] )
                    cv2.circle( image, (cx, cy), 2, color, -1 )

        cv2.imwrite(out, image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return True


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Video anonymisation')
    args.add_argument('-a', '--haar', default=None, type=str, required=True,
                      help='haar cascade model path')
    args.add_argument('-m', '--mask', default=False, action='store_true', 
                      help='mask faces in the videos')
    args.add_argument('-l', '--lbf', default=None, type=str, required=True,
                      help='FacemarkLBF model path')
    args.add_argument('-i', '--input', default=None, type=str, required=True,
                      help='input directory path')
    args.add_argument('-o', '--output', default=None, type=str, required=True,
                      help='path directory to output')
    args_parsed = args.parse_args()
    anonCls = DatasetPrepare(args_parsed)
    anonCls.run(args_parsed.input, args_parsed.output)

# python src/util/anonymise.py -a data/face-model/haarcascade_frontalface_alt2.xml -l data/face-model/lbfmodel.yaml -i data/originalVideos -o data/anonymisedVideos
