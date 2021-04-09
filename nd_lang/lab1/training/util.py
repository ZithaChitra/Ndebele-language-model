""" Function to train a model. """
# from time import time

import importlib
from tensorflow.keras.callbacks import EarlyStopping
from lab1.language_model.datasets.dataset import Dataset
from lab1.language_model.models.base2 import Model
import wandb
import numpy as np
# from wandb.keras import WandbCallback

# early_stop = True




def save_net_artifact(project_name, network_fn):
	"""
	Save artifact of neural net used. For model versioning
	"""
	config = dict(
			input_shape=(13,),
			output_shape=(1),
			layer_size=64,
			dropout_amount=0.2,
			num_layers=3
		)


	with wandb.init(project=project_name, job_type="initialize", config=config) as run:
		config = wandb.config
		
		model = network_fn()

		model_artifact = wandb.Artifact(
            "convnet", type="model",
            description="Simple AlexNet style CNN",
            metadata=dict(config))        

		model.save("initialized_model.keras")
        # ➕ another way to add a file to an Artifact
		model_artifact.new_file("initialized_model.keras")
		wandb.save("initialized_model.keras")

		run.log_artifact(model_artifact)




def save_data_raw_artifact(project_name, data_class):
	""" Save data artifact to wandb for data versioning"""
	data = data_class()
	config=dict(
		name="Blessing",
		surname="Chitakatira"
	)
	with wandb.init(project=project_name ,config=config) as run:
		wandb.log(
			{
				"metric1": 28,
				"metric2": 44
			}
		)
		raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="sklearn.datasets.load_boston",
            metadata={"source": "keras.datasets.mnist",
                      #"size (rows)": [model.dataset.X.shape[0]]
					  })
		with raw_data.new_file("raw" + ".npz", mode="wb") as file:
			np.savez(file, x=data.X, y=data.y)
		run.log_artifact(raw_data)



def save_data_processed_artifact(project_name, data_class):
	data = data_class()
	with wandb.init(project=project_name) as run:
		preprocessed_data = wandb.Artifact(
            "mnist-processed", type="dataset",
            description="sklearn.datasets.load_boston",
            metadata={"source": "keras.datasets.mnist",
                      #"size (rows)": [model.dataset.X.shape[0]]
					  })
		with preprocessed_data.new_file("training" + ".npz", mode="wb") as file:
			np.savez(file, x=data.X_tr, y=data.y_tr)
		run.log_artifact(preprocessed_data)
	

		





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



if __name__ == "__main__":
	""" do something """
	network = "mlp"
	networks_module = importlib.import_module("lab1.language_model.networks.mlp")
	network_fn = getattr(networks_module, network)
	save_net_artifact("test-02", network_fn())
