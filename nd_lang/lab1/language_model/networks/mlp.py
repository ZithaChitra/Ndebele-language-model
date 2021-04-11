""" Define mlp network function. """
from typing import Dict
from tensorflow import keras
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, Flatten
from lab1.language_model.datasets.house_pred import HousingData
from tensorflow.keras import layers

def mlp(
	# input_shape: Tuple[int, ...],
	# output_shape: Tuple[int, ...],
	# layer_size: int = 128,
	# dropout_amount: float = 0.2,
	# num_layers: int = 3, 
	net_config: Dict
)->Model:
	"""
	Creates a simple multi-layer perceptron
	"""
	activation_fn = net_config["hyperparams"]["activation_fn"]
	input_s = net_config["shapes"]["input_shape"]
	output_s = net_config["shapes"]["output_shape"]

	inputs = keras.Input(shape=(input_s,))
	dense = layers.Dense(64, activation="relu")
	x = dense(inputs)
	layer1 = layers.Dense(64, activation=activation_fn)(x)
	layer2 = layers.Dense(64, activation=activation_fn)(layer1)
	outputs = layers.Dense(output_s)(layer2)
	model = keras.Model(inputs=inputs, outputs=outputs, name="house_pred")


	return model


if __name__ == "__main__":
	house_data = HousingData()
	house_data.preprocess_data()
	model = mlp()
	model.summary()
	optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")
	model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])
	model.fit(house_data.X_tr, house_data.y_tr, epochs=500, validation_split=0.2)










