
# YOU ONLY NEED TO GET QUANTIZED .tflite FILE.
# AND TENSOR DATA AUTOMATICALLY GENERATES.

# MODEL ACCURACY MAY BE NOT RELIABLE

# PUT YOUR labels.txt, mobilenet_v2_int8.tflite, apple.jpg(as a test image/auto resize) FILES IN SAME DIRECTORY

# "mobilenet_v2_int8.tflite" from https://github.com/aquapapaya/Post-training-quantization

import os
import tensorflow as tf
import numpy as np

from PIL import Image

model_dir = "mobilenet_v2_int8.tflite"
image_dir = "apple.jpg"

tensordata_output_dirname = "tensor"

original_image_cache = {}

if not os.path.exists("tensor"):
    os.mkdir("tensor")


def preprocess_image(image):
	image = np.array(image)
	# reshape into shape [batch_size, height, width, num_channels]
	img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
	# Use `convert_image_dtype` to convert to floats in the [0,1] range.
	image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
	return image


def load_image(image_url, image_size=256, dynamic_size=False, max_dynamic_size=512):
	"""Loads and preprocesses images."""
	# Cache image file locally.
	if image_url in original_image_cache:
		img = original_image_cache[image_url]
	elif image_url.startswith('https://'):
		# img = load_image_from_url(image_url)
		img = None
		pass
	else:
		fd = tf.io.gfile.GFile(image_url, 'rb')
		img = preprocess_image(Image.open(fd))
	original_image_cache[image_url] = img
	# Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
	img_raw = img
	if tf.reduce_max(img) > 1.0:
		img = img / 255.
	if len(img.shape) == 3:
		img = tf.stack([img, img, img], axis=-1)
	if not dynamic_size:
		img = tf.image.resize_with_pad(img, image_size, image_size)
	elif img.shape[1] > max_dynamic_size or img.shape[2] > max_dynamic_size:
		img = tf.image.resize_with_pad(img, max_dynamic_size, max_dynamic_size)
	return img, img_raw


def reshape_input_image(path, resize=224):
	image, image_raw = load_image(path, resize)
	newimage = np.array(image)
	print(newimage.shape)
	return newimage.reshape((1, 224, 224, 3))


def main():
	with open("./labels.txt", "r") as f:
		lines = f.readlines()
	lines = [line.strip().replace("'", "\"") for line in lines]

	input_images = reshape_input_image(image_dir)

	interpreter = tf.lite.Interpreter(model_dir)
	# tfanalyze = tf.lite.experimental.Analyzer.analyze(model_path=model_dir)
	# print(tfanalyze)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	input_shape = input_details[0]['shape']
	print(f"input shape: {input_shape}")  # [  1 224 224   3]

	# input quantization
	input_images = (input_images / 255.0)
	input_images = input_images.astype('int8')

	# input_images = tf.data.Dataset.from_tensor_slices(x_test).batch(1)
	# output_labels = tf.data.Dataset.from_tensor_slices(y_test).batch(1)
	# print(f"data type of input_images: {type(input_images)}")
	# print(input_images.get_single_element(1))
	# interpreter.allocate_tensors()

	interpreter.set_tensor(input_details[0]['index'], input_images)
	interpreter.invoke()
	for e in interpreter.get_tensor_details():
		if not 'depthwise/depthwise' in e['name']:
			# interpreter.invoke()
			tensor = interpreter.get_tensor(e["index"])
			tensor_name = e['name'].replace(';', '_').replace('/', '_').replace(':', '_')
			np.save(os.path.join(tensordata_output_dirname, f"{tensor_name}"), tensor)

	output_result = interpreter.get_tensor(output_details[0]['index'])[0]
	inf_result_idx = np.argmax(output_result) - 1
	print(lines[inf_result_idx])

	# print(interpreter.get_tensor_details())
	#
	# for tensor_details in interpreter.get_tensor_details():
	# 	if 'expanded_conv' in tensor_details['name'] \
	# 		or 'Conv2d' in tensor_details['name'] \
	# 		or 'AvgPool' in tensor_details['name']:
	# 		tensor_name = tensor_details['name'].replace(';', '_').replace('/', '_').replace(':', '_')
	# 		# interpreter.allocate_tensors()
	# 		tensor = interpreter.get_tensor(tensor_details["index"])
	# 		np.save(os.path.join(tensordata_output_dirname, f"{tensor_name}"), tensor)

	# output_result = interpreter.get_tensor(output_details[0]['index'])[0]
	# print(output_result)
	# inf_result = np.argmax(output_result) - 1
	# print(lines[inf_result])

	# for xt, yt in zip(input_images, output_labels):
	# 	cnt += 1
	# 	interpreter.set_tensor(input_details[0]['index'], xt)
	# 	interpreter.invoke()
	# 	output_data = interpreter.get_tensor(output_details[0]['index'])
	# 	print(f"\riteration: {cnt:6} exact: {evaluation:6}   inferenced: {output_data[0]} answer: {np.array(yt[0])}",
	# 		  end='')
	# 	# print(output_data[0])
	# 	# print("yt to np")
	# 	# print(np.array(yt[0]))
	# 	if np.argmax(output_data[0]) == np.argmax(np.array(yt[0])):
	# 		evaluation += 1
	#
	# print(f"\nquantized model acccuracy: {evaluation / cnt}")
	#
	# # Save first intermediate output tensors
	# xt = list(input_images.as_numpy_iterator())[0]
	# interpreter.set_tensor(input_details[0]['index'], xt)
	# interpreter.invoke()

	# for tensor_details in interpreter.get_tensor_details():
	# 	if 'Conv2D' in tensor_details['name'] \
	# 			or 'conv2d' in tensor_details['name'] \
	# 			or 'MaxPool2D' in tensor_details['name'] \
	# 			or 'maxpool2d' in tensor_details['name']:
	# 		tensor_name = tensor_details['name'].replace(';', '_').replace('/', '_').replace(':', '_')
	# 		tensor = interpreter.get_tensor(tensor_details["index"])
	# 		np.save(os.path.join(intermediate_output_dirname, f" {tensor_name}"), tensor)
	#
	# print(f"intermediate layer outputs are saved at {intermediate_output_dirname}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	main()

