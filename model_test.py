import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

src_dir = os.path.join("src")
sys.path.insert(0, src_dir)

from model import VTNBuilder

# model = VTNVITBuilder(backbone="vit_b16", HIDDEN_DIM=768, NUM_ATTENTION_HEADS=8)

model = VTNBuilder(temporal="VTNLongformerLayer", input_shape=[3, 16, 224, 224])

test = tf.random.uniform((1, 3, 16, 224, 224))   # B, C, F, H, W
frame_idx = np.arange(0, 1*16).reshape((1, 16))
out = model([test, frame_idx])
print(out.shape)
model.summary(line_length=400)
table = pd.DataFrame(columns=["Name", "Type", "Output Shape", "Param #"])
for layer in model.layers:
    table = table.append(
        {
            "Name": layer.name, 
            "Type": layer.__class__.__name__, 
            "Output Shape": layer.output_shape,
            "Param #": layer.count_params()
        }, ignore_index=True)
print(table.to_latex(index=False))
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
model.save_weights("saved/pcs-vtn-test")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
converter.experimental_new_converter = True
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# == SELECT_TF_OPS ==
# Flex ops: FlexEinsum, FlexEqual, FlexErf, FlexRoll
# Save the model.
with open('saved/pcs-vtn-test.tflite', 'wb') as f:
    f.write(tflite_model)
