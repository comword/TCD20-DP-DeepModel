import json
import random
import cv2
import collections
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from utils import initializer
from . import transform


class FrameDataLoader(tf.keras.utils.Sequence):
    @initializer
    def __init__(self, data_path, batch_frame, batch_second, num_clips=12, shuffle=True):
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
        if shuffle:
            self.shuffle_res = random.sample(
                range(self.__len__()), self.__len__())

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
        len_videos = len(list(self.files))
        vid = list(self.files)[idx % len_videos]
        label = int(self.labels.loc[vid.split('/')[0]]['label'])
        fps = int(self.labels.loc[vid.split('/')[0]]['fps'])
        start_idx, end_idx = self.get_start_end_idx(len(self.files[vid]),
                                                    fps * self.batch_second,
                                                    idx // len_videos,
                                                    self.num_clips)
        frame_idx = np.linspace(
            start_idx, end_idx, self.batch_frame).astype(int)
        frame_idx = np.clip(frame_idx, 0, len(self.files[vid])-1)

        frames = []
        for i in frame_idx:
            frames.append(self.files[vid][i][1])
        return frames, frame_idx-np.min(frame_idx), label

    def __call__(self):
        if self.shuffle:
            for i in self.shuffle_res:
                yield self.__getitem__(i)
        else:
            for i in range(self.__len__()):
                yield self.__getitem__(i)


def dataloader(imgs, frame_idx, label, resize_to, resolution, mode='train'):
    imgs = imgs.numpy()
    img_shape = cv2.imread(imgs[0].decode("utf-8")).shape  # H, W, C
    frames = np.zeros(
        (imgs.shape[0], img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    fid = 0
    for i in imgs:
        raw_data = cv2.imread(i.decode("utf-8"))
        rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB) / 255.0
        frames[fid] = rgb_data
        fid += 1

    if mode == 'train':
        frames, _ = transform.random_short_side_scale_jitter_list(
            images=frames,
            min_size=resize_to[0],
            max_size=resize_to[1]
        )
        frames = np.transpose(np.asarray(frames), (0, 3, 1, 2))  # F, C, H, W
        frames, _ = transform.random_crop(frames, resolution)
        frames = transform.horizontal_flip(0.5, frames, order="CHW")
        frames = np.transpose(frames, (1, 0, 2, 3))  # C, F, H, W
    else:
        frames, _ = transform.random_short_side_scale_jitter_list(
            frames, resolution, resolution
        )
        # FHWC
        frames = np.transpose(np.asarray(frames), (0, 3, 1, 2))  # F, C, H, W
        frames, _ = transform.uniform_crop(frames, resolution, 1)
        frames = np.transpose(frames, (1, 0, 2, 3))  # C, F, H, W

    return frames, frame_idx, label


class FrameDataLoaderTF:

    def __init__(self, *args, batch_size=15, resolution=224, resize_to=[0.8, 1.2],
                 num_clips=12, shuffle=True, validation_split=0.1, **kwargs):
        self.batch_size = batch_size
        self.loader = FrameDataLoader(
            *args, shuffle=shuffle, num_clips=num_clips, **kwargs)
        types = (tf.string, tf.int32, tf.int32)
        ds = tf.data.Dataset.from_generator(self.loader, output_types=types)
        self.ds = ds
        if validation_split > 0:
            self.split_validation = True
            self.val_size = int(validation_split * self.loader.__len__())
            self.train_dataset = ds.skip(self.val_size).map(lambda imgs, frame_idx, label: tf.py_function(dataloader,
                                                                                                          inp=[
                                                                                                              imgs, frame_idx, label, resize_to, resolution, 'train'],
                                                                                                          Tout=[tf.float32, tf.int32, tf.int32]), num_parallel_calls=16).batch(batch_size).prefetch(batch_size)
            self.val_dataset = ds.take(self.val_size).map(lambda imgs, frame_idx, label: tf.py_function(dataloader,
                                                                                                        inp=[
                                                                                                            imgs, frame_idx, label, resize_to, resolution, 'val'],
                                                                                                        Tout=[tf.float32, tf.int32, tf.int32]), num_parallel_calls=16).batch(batch_size).prefetch(batch_size)
        else:
            self.split_validation = False
            self.val_size = 0
            self.train_dataset = ds.map(lambda imgs, frame_idx, label: tf.py_function(dataloader,
                                                                                      inp=[
                                                                                          imgs, frame_idx, label, resize_to,
                                                                                          mean_norm, std_norm, resolution, 'train'],
                                                                                      Tout=[tf.float32, tf.int32, tf.int32]), num_parallel_calls=16).batch(batch_size).prefetch(batch_size)

    def hasSplitValidation(self):
        return self.split_validation

    def getFullDataset(self):
        return self.train_dataset

    def getSplitDataset(self):
        return self.train_dataset, self.val_dataset

    def getTrainLen(self):
        return (self.loader.__len__() - self.val_size) // self.batch_size

    def getValLen(self):
        return self.val_size // self.batch_size
