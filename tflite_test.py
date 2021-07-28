import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter("saved/pcs-vtn.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_shape_1 = input_details[1]['shape']
input_data_1 = np.array(np.random.random_sample(
    input_shape_1), dtype=np.float32)

print(input_details)

# [{'name': 'input_1', 'index': 0, 'shape': array([1,   3,  16, 224, 224], dtype=int32), 'shape_signature': array([-1,   3,  16, 224, 224], dtype=int32), 'dtype': < class 'numpy.float32' > , 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'input_2', 'index': 1, 'shape': array([1, 16], dtype=int32), 'shape_signature': array([-1, 16], dtype=int32), 'dtype': < class 'numpy.float32' > , 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.set_tensor(input_details[1]['index'], input_data_1)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
