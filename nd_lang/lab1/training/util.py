""" Function to train a model. """
# from time import time

from tensorflow.keras.callbacks import EarlyStopping
from lab1.language_model.datasets.dataset import Dataset
from lab1.language_model.models.base2 import Model
import wandb
# from wandb.keras import WandbCallback

# early_stop = True




def save_net_artifact(project_name, network, config):
	"""
	Neural Net used artifact. For model versioning
	"""
	with wandb.init(project=project_name, job_type="initialize", config=config) as run:
		config = wandb.config
		
		model = network

		model_artifact = wandb.Artifact(
            "convnet", type="model",
            description="Simple AlexNet style CNN",
            metadata=dict(config))        

		model.save("initialized_model.keras")
        # ➕ another way to add a file to an Artifact
		model_artifact.add_file("initialized_model.keras")
		wandb.save("initialized_model.keras")

		run.log_artifact(model_artifact)




def save_data_artifact(dataset):
	""" Save data artifact to wandb for data versioning"""





def train_model(model: Model,
			dataset: Dataset,
			epochs: int, 
			# batch_size: int,
			use_wandb: bool = True) -> Model:
	
	""" Train model. """
	callbacks = []

	# if early_stop:
	# 	early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=1, mode="auto")
	# 	callbacks.append(early_stopping)

	# if use_wandb:
	# 	callbacks.append(WandbCallback)

	# model.network.summary()
	# t = time()
	model.fit(dataset=dataset, 
				# batch_size=batch_size, 
				epochs=epochs, 
				callbacks=callbacks)
	# print("Training took {:2f} s".format(time() - 1))

	
	return Model



