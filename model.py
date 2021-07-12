import os
import sys
import tensorflow as tf
import numpy as np

src_dir = os.path.join("src")
sys.path.insert(0, src_dir)

from model.vtn import VTNBuilder

model = VTNBuilder()

test = [tf.random.uniform((1, 3, 25, 224, 224)), np.arange(
    0, 1*25).reshape((1, 25))]  # B, C, F, H, W
out = model(test)
print(out.shape)
model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# efficientnetb0 (Functional)  (None, 7, 7, 1280)        4049571
# _________________________________________________________________
# sequential (Sequential)      (25, 384)                 491904
# _________________________________________________________________
# vtn_distil_bert_layer (VTNDi multiple                  14532608
# _________________________________________________________________
# sequential_1 (Sequential)    (1, 16)                   154000
# =================================================================
# Total params: 19,228,467
# Trainable params: 19,186,444
# Non-trainable params: 42,023
# _________________________________________________________________

# model.compile(optimizer='sgd', loss='mean_squared_error')
# def _input_fn():
#   img = np.random.rand(3, 25, 224, 224)
#   frame_idx = np.arange(0, 1*25).reshape((1, 25))

#   labels = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.005,
#               0.003, 0.001, 0.011, 0, 0, 0, 0, 0, 0, 0])

#   def generator():
#     for s1, s2, l in zip(img, frame_idx, labels):
#       yield {"img": s1, "frame_idx": s2}, l

#   dataset = tf.data.Dataset.from_generator(generator, output_types=({"img": tf.float32, "frame_idx": tf.int32}, tf.float32))
#   dataset = dataset.batch(1)
#   return dataset

# model.fit(_input_fn(), epochs=2)
model.save_weights("saved/pcs-vtn")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
converter.experimental_new_converter = True
# converter.allow_custom_ops = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# == SELECT_TF_OPS ==
# tf.Erf {device = ""}
# tf.FusedBatchNormV3 {data_format = "NHWC", device = "", epsilon = 1.000000e-03 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = true}

# Save the model.
with open('saved/pcs-vtn.tflite', 'wb') as f:
    f.write(tflite_model)
