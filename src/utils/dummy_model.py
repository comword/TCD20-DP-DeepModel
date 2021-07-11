import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=20, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='softmax')
])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics='acc')
X = np.arange(0, 16)
X = X.reshape((16, 1))
y = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.005,
              0.003, 0.001, 0.011, 0, 0, 0, 0, 0, 0, 0])
print('Shape of X is ', X.shape)
print('Shape of y is', y.shape)
model.fit(x=X, y=y, epochs=5)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('data/dummy.tflite', 'wb') as f:
    f.write(tflite_model)
