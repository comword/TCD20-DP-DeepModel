import os
import sys
import tensorflow as tf
from PIL import Image


src_dir = os.path.join("src")
sys.path.insert(0, src_dir)

from dataloader import FrameDataLoader

batch_size = 1
dl = FrameDataLoader("data/anonymisedVideos", [224, 224], 15, 5, 12)

types = (tf.float32, tf.int32, tf.int32)
ds = tf.data.Dataset.from_generator(dl, output_types=types)
ds = ds.batch(batch_size)

# X, frame_idx, y = next(iter(ds))
# print(X.shape, frame_idx, y)

X, frame_idx, y = dl.__getitem__(0)