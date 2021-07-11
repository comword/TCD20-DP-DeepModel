import os, sys
import tensorflow as tf

src_dir = os.path.join("src")
sys.path.insert(0, src_dir)

from model.vtn import VTNBuilder

model = VTNBuilder()

test = [tf.random.uniform((2, 3, 25, 224, 224)), tf.random.uniform((2, 25))]  # B, C, F, H, W
out = model(test)
print(out.shape)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('data/pcs-vtn.tflite', 'wb') as f:
    f.write(tflite_model)