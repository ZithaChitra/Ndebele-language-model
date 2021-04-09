""" Script to run an experiment """
import importlib
# from typing import Dict
# import os
import click

# from lab1.training.util import train_model
from lab1.training.util import save_net_artifact, save_data_raw_artifact, save_data_processed_artifact
import wandb
from wandb.keras import WandbCallback
import numpy as np


DEFAULT_TRAIN_ARGS = {"batch_size":64, "epochs":16}


@click.command()
@click.argument("dataset", default="HousingData")
@click.argument("network", default="mlp")
@click.argument("model", default="Model")
@click.option("--proj-name", default="nd_lang")
@click.option("--epoch", default=10)
@click.option("--train-args", default=DEFAULT_TRAIN_ARGS)
def run_experiment(dataset, network, model, proj_name, epoch, train_args):

	print(f"Running experiment with network '{network}' and dataset '{dataset}''")
	datasets_module = importlib.import_module("lab1.language_model.datasets.house_pred")
	dataset_class_ = getattr(datasets_module, dataset)
	# dataset_args = experiment_config.get("dataset_args", {})


	models_module = importlib.import_module("lab1.language_model.models.base2")
	model_class_ = getattr(models_module, model)

	networks_module = importlib.import_module("lab1.language_model.networks.mlp")
	network_fn = getattr(networks_module, network)
	
	
	# network_args = experiment_config.get("network_args", {})

	# mlflow.set_tracking_uri("sqlite:///mlruns.db")
	model = model_class_(dataset_cls=dataset_class_, network_fn=network_fn)
	# input_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, 13), name="house_attribs")])
	# output_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, 1), name="predicted house price")])
	# signature = ModelSignature(inputs=input_schema, outputs=output_schema)
	# input_example = np.array([[1., 2.5, 3. , 1.7, 2.1, 1.3, .5, .75, .89, 1.9, 2.15, 2.2, .6]])
	# mlflow.pyfunc.save_model(path="my_model", python_model=model, signature=signature, input_example=input_example )

	config = dict(
		dataset = dataset,
		network = network,
		model = model,
		epoch = epoch,
		train_args = train_args
	)

	net_config = dict(
			input_shape=(13,),
			output_shape=(1),
			layer_size=64,
			dropout_amount=0.2,
			num_layers=3
		)
	
	save_net_artifact(project_name=proj_name, network_fn=network_fn)
	save_data_raw_artifact(project_name=proj_name, data_class=dataset_class_)
	save_data_processed_artifact(project_name=proj_name, data_class=dataset_class_)
	with wandb.init(project=proj_name, config=config) as run:
		config = wandb.config
        
		

		model.fit(dataset=config.dataset, callbacks=[WandbCallback()])


	# model_ = train_model(
	# 		model,
	# 		dataset,
	# 		epoch
	# 	)




if __name__ == "__main__":
	run_experiment()
	

