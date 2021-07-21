import os
import random
import numpy as np
import tensorflow as tf
from utils import initializer
from iopath.common.file_io import g_pathmgr
from . import decoder as decoder
from . import video_container as container
from . import transform


class VideoDataLoader(tf.keras.utils.Sequence):

    @initializer
    def __init__(self, data_path, batch_frame, sampling_rate, mode="train",
                 resize_to=[256, 320], resolution=224, mean_norm=[0.45, 0.45, 0.45],
                 std_norm=[0.225, 0.225, 0.225], num_clips=5, shuffle=True):
        self._video_meta = {}
        self._num_retries = 10
        self._num_clips = num_clips
        self.num_frames = batch_frame
        if self.mode in ["val", "test"]:
            self.force_all_video = True
            # self.sampling_rate = 1
        else:
            self.force_all_video = False
            self.sampling_rate = sampling_rate

        self._construct_loader()
        if shuffle:
            self.shuffle_res = random.sample(
                range(self.__len__()), self.__len__())

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.data_path, "{}.csv".format(self.mode)
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(" "))
                    == 2
                )
                path, label = path_label.split(" ")
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.data_path, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )

    def __len__(self):
        return self.num_videos

    @property
    def num_videos(self):
        return len(self._path_to_videos)

    def get_random_sampling_rate(self, long_cycle_sampling_rate, sampling_rate):
        """
        When multigrid training uses a fewer number of frames, we randomly
        increase the sampling rate so that some clips cover the original span.
        """
        if long_cycle_sampling_rate > 0:
            assert long_cycle_sampling_rate >= sampling_rate
            return random.randint(sampling_rate, long_cycle_sampling_rate)
        else:
            return sampling_rate

    def __getitem__(self, index):
        if self.mode in ["train", "val", "all"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.resize_to[0]
            max_scale = self.resize_to[1]
            crop_size = self.resolution
            if self.mode in ["val", "all"]:
                # supporting full video evaluation:
                # spatial_sample_index=1 to take only the center
                # The testing is deterministic and no jitter should be performed.
                # min_scale, max_scale, and crop_size are expect to be the same.
                # temporal_sample_index = -1  # this can be random - in the end we take [0,inf]
                spatial_sample_index = 1
                min_scale = self.resolution
                max_scale = self.resolution
                crop_size = self.resolution
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // 1
                # // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = 1
            min_scale = self.resolution
            max_scale = self.resolution
            crop_size = self.resolution
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = self.get_random_sampling_rate(0, self.sampling_rate)
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index], True, "pyav"
                )
            except Exception as e:
                print(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                print(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames, frames_index = decoder.decode(
                video_container,
                sampling_rate,
                self.num_frames,
                temporal_sample_index,
                # self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                1,
                video_meta=self._video_meta[index],
                # target_fps=self.cfg.DATA.TARGET_FPS,
                target_fps=25,
                max_spatial_scale=min_scale,
                force_all_video=self.force_all_video
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                print(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            # frames = utils.tensor_normalize(
            #     frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            # )
            frames = frames / 255.0
            frames = np.transpose(frames, (0, 3, 1, 2))  # F, C, H, W
            frames = transform.color_normalization(
                frames, self.mean_norm, self.std_norm)
            frames = np.transpose(frames, (0, 2, 3, 1))  # F, H, W, C
            # Perform data augmentation.
            frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=True,
                inverse_uniform_sampling=False,
            )
            frames = np.transpose(frames, (3, 0, 1, 2))  # C, F, H, W
            label = self._labels[index]
            # frames = [frames, frames_index]
            return frames, frames_index, label
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def spatial_sampling(self,
                         frames,
                         spatial_idx=-1,
                         min_scale=256,
                         max_scale=320,
                         crop_size=224,
                         random_horizontal_flip=True,
                         inverse_uniform_sampling=False,
                         ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
            inverse_uniform_sampling (bool): if True, sample uniformly in
                [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
                scale. If False, take a uniform sample from [min_scale,
                max_scale].
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter_list(
                images=frames,
                min_size=min_scale,
                max_size=max_scale
                # inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = transform.random_crop_list(
                frames, crop_size, order="HWC")
            if random_horizontal_flip:
                frames = transform.horizontal_flip(
                    0.5, np.asarray(frames), order="HWC")
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale and max_scale are expect to be the same.
            assert min_scale == max_scale
            frames, _ = transform.random_short_side_scale_jitter_list(
                frames, min_scale, max_scale
            )
            # FHWC
            frames = np.transpose(np.asarray(
                frames), (0, 3, 1, 2))  # F, C, H, W
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
            frames = np.transpose(frames, (0, 2, 3, 1))
        return frames

    def __call__(self):
        if self.shuffle:
            for i in self.shuffle_res:
                yield self.__getitem__(i)
        else:
            for i in range(self.__len__()):
                yield self.__getitem__(i)


class VideoDataLoaderTF:

    def __init__(self, *args, batch_size=15, resolution=224, resize_to=[256, 320], num_clips=5, mode='splitted',
                 mean_norm=[0.45, 0.45, 0.45], std_norm=[0.225, 0.225, 0.225], shuffle=True, validation_split=0.1, **kwargs):
        self.batch_size = batch_size
        self.mode = mode
        types = (tf.float32, tf.int32, tf.int32)
        if mode == 'splitted':
            self.split_validation = True
            self.train_loader = VideoDataLoader(*args, resolution=resolution, resize_to=resize_to, num_clips=num_clips,
                                                mean_norm=mean_norm, std_norm=std_norm, mode="train", shuffle=shuffle, **kwargs)
            self.train_ds = tf.data.Dataset.from_generator(
                self.train_loader, output_types=types).batch(batch_size).prefetch(batch_size)
            self.val_loader = VideoDataLoader(*args, resolution=resolution, resize_to=resize_to, num_clips=num_clips,
                                              mean_norm=mean_norm, std_norm=std_norm, mode="val", shuffle=shuffle, **kwargs)
            self.val_ds = tf.data.Dataset.from_generator(
                self.val_loader, output_types=types).batch(batch_size).prefetch(batch_size)
        elif mode == 'all':
            self.all_loader = VideoDataLoader(*args, resolution=resolution, resize_to=resize_to, num_clips=num_clips,
                                              mean_norm=mean_norm, std_norm=std_norm, mode="all", shuffle=shuffle, **kwargs)
            self.all_ds = tf.data.Dataset.from_generator(
                self.all_loader, output_types=types)
            if validation_split > 0:
                self.split_validation = True
                self.val_size = int(validation_split *
                                    self.all_loader.__len__())
                self.train_ds = self.all_ds.skip(self.val_size).batch(
                    batch_size).prefetch(batch_size)
                self.val_ds = self.all_ds.take(self.val_size)
            else:
                self.split_validation = False
                self.val_size = 0
                self.train_ds = tf.data.Dataset.from_generator(
                    self.all_loader, output_types=types).batch(batch_size).prefetch(batch_size)
        if shuffle:
            self.train_ds = self.train_ds.shuffle(batch_size)

    def hasSplitValidation(self):
        return self.split_validation

    def getFullDataset(self):
        return self.train_ds

    def getSplitDataset(self):
        return self.train_ds, self.val_ds

    def getTrainLen(self):
        if self.mode == 'splitted':
            return self.train_loader.__len__() // self.batch_size
        else:
            return (self.all_loader.__len__() - self.val_size) // self.batch_size

    def getValLen(self):
        if not self.hasSplitValidation():
            return 0
        if self.mode == 'splitted':
            return self.val_loader.__len__() // self.batch_size
        else:
            return self.val_size // self.batch_size
