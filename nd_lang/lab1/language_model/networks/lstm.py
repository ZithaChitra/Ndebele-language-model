""" Defines an LSTM RNN function """

from typing import Tuple

import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow import keras
from tensorflow.keras import layers



def lstm(
	input_shape: Tuple[int, ...],
	output_shape: Tuple[int, ...],
	layer_size: int = 20,
	dropout_amount: float = 0.2,
	num_layers: int = 3
	)->Model:
	"""
	Creates a multi-layer LSTM with dropout between the layers. 
	"""


	return model



























model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and 
# output embedding dimension of size 64
model.add(layers.Embedding(input_dim=1000, output_dim=64))

#Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128, return_state=True)) # has two state tensors

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()


# to configure the intial state of the layer, just call the layer
# with additional keyword argument "initial_state"
# the shape of the state needs to match the unit size of the
# layer

encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None,))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(
	encoder_input
	)

# Return states in addition to output
output, state_h, state_c = layers.LSTM(64, return_state=True, name="encoder" )(
	encoder_input
	)

encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None,))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(
	decoder_input
	)

# Pass the 2 states to a new LSTM layer, as initial state
decoder_output = layers.LSTM(64, name="decoder")(
	decoder_embedded, initial_state=encoder_state
	)

output = layers.Dense(10)(decoder_output)

model = keras.Model([encoder_input, decoder_input], output)
model.summary()
