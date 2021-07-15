import json
import random
import cv2
import collections
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from utils import initializer
from skimage.transform import resize
from .transform import random_crop, color_normalization


class FrameDataLoader(tf.keras.utils.Sequence):
    @initializer
    def __init__(self, data_path, resolution, batch_frame, batch_second,
                 num_clips=15, resize_fac=[0.8, 1.2], mean_norm=[0.45, 0.45, 0.45], std_norm=[0.225, 0.225, 0.225]):
        self.data_path = Path(data_path)
        with open(str(self.data_path / "types.json"), "r") as f:
            self.cls_types = json.load(f)
        self.labels = pd.read_csv(
            self.data_path / "videos.csv", index_col="name")

        file_list = list(self.data_path.glob("*.jpg"))
        temp_files = dict(zip(map(lambda x: x.stem, file_list),
                              map(lambda x: str(x), file_list)))

        files = {}
        for (k, v) in temp_files.items():
            ksplit = k.split("_")
            video_name, frame_id = "_".join(ksplit[:-1]), int(ksplit[-1])
            if not video_name in files:
                files[video_name] = {}
            files[video_name][frame_id] = v

        temp_files = {}
        for k in files.keys():
            files[k] = sorted(files[k].items())
            video_fps = int(self.labels.loc[k]['fps'])
            video_size = len(files[k])
            if(video_size > 12 * video_fps):
                split_num = video_size // (12 * video_fps)
                for x in range(0, split_num):
                    start_idx, end_idx = self.get_start_end_idx(
                        video_size, 12 * video_fps, x, split_num)
                    temp_files["%s/%d" %
                               (k, x)] = files[k][round(start_idx):round(end_idx)]
            else:
                temp_files[k] = files[k]
        self.files = collections.OrderedDict(temp_files)

    def get_start_end_idx(self, video_size, clip_size, clip_idx, num_clips):
        delta = max(video_size - clip_size, 0)
        if clip_idx == -1:
            start_idx = random.uniform(0, delta)
        else:
            start_idx = delta * clip_idx / num_clips
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx

    def __len__(self):
        return (len(list(self.files.keys())) * self.num_clips)

    def __getitem__(self, idx):
        vid = list(self.files)[idx // self.num_clips]
        label = int(self.labels.loc[vid.split('/')[0]]['label'])
        fps = int(self.labels.loc[vid.split('/')[0]]['fps'])
        start_idx, end_idx = self.get_start_end_idx(len(self.files[vid]),
                                                    fps * self.batch_second,
                                                    idx % self.num_clips,
                                                    self.num_clips)
        frame_idx = np.linspace(
            start_idx, end_idx, self.batch_frame).astype(int)
        frame_idx = np.clip(frame_idx, 0, len(self.files[vid])-1)
        img_shape = cv2.imread(self.files[vid][0][1]).shape  # H, W, C
        frames = np.zeros(
            (self.batch_frame, img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
        fid = 0
        for i in frame_idx:
            raw_data = cv2.imread(self.files[vid][i-1][1])
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB) / 255.0
            frames[fid] = rgb_data
            fid += 1
        
        frames = np.transpose(frames, (0, 3, 1, 2))  # F, C, H, W
        frames = color_normalization(frames, self.mean_norm, self.std_norm)
        # resize data augment
        resize_factor = random.uniform(self.resize_fac[0], self.resize_fac[1])
        resized = np.zeros((self.batch_frame, frames.shape[1], int(
            frames.shape[2]*resize_factor), int(frames.shape[3]*resize_factor)))
        for n, i in enumerate(frames):
            resized[n, :, :, :] = resize(
                frames[n, :, :, :], resized.shape[1:], anti_aliasing=True)
        frames = resized

        # random crop data augment
        img_shape = frames.shape
        if img_shape[2] != img_shape[3]:
            frames = random_crop(frames, min(
                img_shape[2], img_shape[3]) * resize_factor)
        # if img_shape[2] > self.resolution[0] and img_shape[3] > self.resolution[1]:
        #     pass

        # final resize
        resized = np.zeros((self.batch_frame, frames.shape[1], self.resolution[0], self.resolution[1]))
        for n, i in enumerate(frames):
            resized[n, :, :, :] = resize(
                frames[n, :, :, :], resized.shape[1:], anti_aliasing=True)
        frames = np.transpose(resized, (0, 2, 3, 1)) # F, H, W, C
        return frames, frame_idx, label

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
