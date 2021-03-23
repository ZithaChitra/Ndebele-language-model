""" Function to train a model. """
from time import time

from tensoeflow.keras.callbacks import EarlyStopping, Callback



from text_recognizer.datasets.dataset import Dataset
from text_recognizer.model.base import Model

EarlyStopping = True


def train_model(model: Model, dataset: Dataset, epoch: int, batch_size: int,
				 use_wandb: bool = True) -> Model:
	
	""" Train model. """
	callbacks = []

	if EarlyStopping:
		early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=3, verbose=1, mode="auto")
		callbacks.append(early_stopping)


	model.network.summary()

	t = time()
	_history = model.fit(dataset=dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
	print("Training took {:2f} s".format(time() - 1))

	
	return Model
