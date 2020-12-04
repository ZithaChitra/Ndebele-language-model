"""
Converts data into byte strings that are stored in TFRecord files

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


 def serialize_example(list_of_features):
 	"""
  	Creates a tf.train.Example message ready to be written to a file.
  	"""
  	# Create a dictionary mapping the feature name to the tf.train.Example-compatible
  	# data type.

