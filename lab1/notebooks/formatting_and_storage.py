"""
Methods for converting data into byte strings and for writting serialized
data into TFRecord files.

"""

import tensorflow as tf
import numpy as np


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# The above three functions return proto messages that can
# be serialized to a binary-string using the .SerializaToString method


 def serialize_example(feature0, feature1, feature2, feature3):
 	"""
  	Creates a tf.train.Example message ready to be written to a file.
  	"""
  	# Create a dictionary mapping the feature name to the tf.train.Example-compatible
  	# data type.
  	feature = {
  		"feature0": _int64_feature(feature0),
  		"feature1": _int64_feature(feature1),
  		"feature2":, _bytes_feature(feature2)
  		"feature3":, _float_feature(feature3)
  	}

  	proto_message = tf.train.Example(features=tf.train.Features(
  										feature=feature))
  	return proto_message.SerializaToString()



def tf_serialize_example(f0, f1, f2, f3):
	"""
	Wrapper for the serialize_example function so it can be used as 
	as argument to .map() Dataset function
	"""
	tf_string = tf.py_function(
		serialize_example,
		(f0, f1, f2, f3), # pass these args to the above function
		tf.string # the return type
		)
	return tf.reshape(tf_string, ())



# def generator():
#   for features in features_dataset:
#     yield serialize_example(*features)


# serialized_features_dataset = tf.data.Dataset.from_generator(
#     generator, output_types=tf.string, output_shapes=())


# Using TFRecordDataset can be useful for standardizing input data
# optimizing perfomance

def write_to_tfrecord(file_name, serialized_features_dataset):
	"""
	Wrties data to a TFRecord file.

	"""
	filename = file_name
	writer = tf.data.experimental.TFRecordWriter(filename)
	writer.write(serialized_features_dataset)
	return


def read_from_tfrecord(filenames):
	raw_datasets = []
	for file in filenames:
		raw_datasets.append(tf.data.TFRecordDataset)
	return raw_datasets



# To parse a "tf.train.Example" proto you need to provide
# a feature_description mapping feature names to their shape
# and type.

feature_description = {
	"feature0": tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"feature1": tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"feature2":tf.io.FixedLenFeature([], tf.string, default_value=""),
	"feature3":tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
	# parse the input "tf"




