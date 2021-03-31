""" Function to train a model. """
# from time import time

from tensorflow.keras.callbacks import EarlyStopping



from lab1.language_model.datasets.dataset import Dataset
from lab1.language_model.models.base2 import Model

early_stop = True


def train_model(model: Model,
			dataset: Dataset,
			epochs: int, 
			# batch_size: int,
			use_wandb: bool = False) -> Model:
	
	""" Train model. """
	callbacks = []

	if early_stop:
		early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=1, mode="auto")
		callbacks.append(early_stopping)

	model.network.summary()
	# t = time()
	model.fit(dataset=dataset, 
				# batch_size=batch_size, 
				epochs=epochs, 
				callbacks=callbacks)
	# print("Training took {:2f} s".format(time() - 1))

	
	return Model



