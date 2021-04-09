""" Script to run an experiment """
import importlib
# from typing import Dict
# import os
import click
# import mlflow

# from lab1.training.util import train_model
from lab1.training.util import save_net_artifact
import wandb
from wandb.keras import WandbCallback

# from mlflow.models.signature import ModelSignature
# from mlflow.types.schema import TensorSpec, Schema
# import numpy as np


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
	dataset = dataset_class_()
	# dataset.load_or_generate_data()

	models_module = importlib.import_module("lab1.language_model.models.base2")
	model_class_ = getattr(models_module, model)

	networks_module = importlib.import_module("lab1.language_model.networks.mlp")
	network_fn = getattr(networks_module, network)
	# save_net_artifact(project_name=proj_name, network=network_fn())
	
	# network_args = experiment_config.get("network_args", {})

	# mlflow.set_tracking_uri("sqlite:///mlruns.db")
	model = model_class_(dataset_cls=dataset_class_, network_fn=network_fn)
	# input_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, 13), name="house_attribs")])
	# output_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, 1), name="predicted house price")])
	# signature = ModelSignature(inputs=input_schema, outputs=output_schema)
	# input_example = np.array([[1., 2.5, 3. , 1.7, 2.1, 1.3, .5, .75, .89, 1.9, 2.15, 2.2, .6]])
	# mlflow.pyfunc.save_model(path="my_model", python_model=model, signature=signature, input_example=input_example )

	# with mlflow.start_run():
	# 	# mlflow.log_param("dataset", dataset)
	# 	# mlflow.log_param("network", network)
	# 	# mlflow.log_param("model", model)
	
	# 	model = model_class_(dataset_cls=dataset_class_, network_fn=network_fn)
		
	# 	# mlflow.keras.autolog()
	# 	# model_ = train_model(
	# 	# 	model,
	# 	# 	dataset,
	# 	# 	epochs=epoch,
	# 	# 	# batch_size=experiment_config["train_args"]["batch_size"]
	# 	# )
	# 	mlflow.pyfunc.save_model(path="my_model", python_model=model )

	config = dict(
		dataset = dataset,
		network = network,
		model = model,
		epoch = epoch,
		train_args = train_args
	)
	
	with wandb.init(project=proj_name, config=config):
		config = wandb.config
		model.fit(dataset=config.dataset, callbacks=[WandbCallback()])




	# model_ = train_model(
	# 		model,
	# 		dataset,
	# 		epoch
	# 	)
	# callbacks = []
	# model.fit(dataset=dataset)






if __name__ == "__main__":
	# args = get_args()
	# print(args["dataset"], args["network"], args["model"], args["epoch"], args["train_args"])
	# run_experiment("HousingData", "mlp", "Model", 10, DEFAULT_TRAIN_ARGS )
	run_experiment()
	

