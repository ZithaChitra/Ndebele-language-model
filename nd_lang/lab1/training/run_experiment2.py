""" Script to run an experiment """
import importlib
# from typing import Dict
# import os
import click

from lab1.training.util import train_model

DEFAULT_TRAIN_ARGS = {"batch_size":64, "epochs":16}



@click.command()
@click.argument("dataset", default="HousingData")
@click.argument("network", default="mlp")
@click.argument("model", default="Model")
@click.option("--epoch", default=10)
@click.option("--train-args", default=DEFAULT_TRAIN_ARGS)
def run_experiment(dataset, network, model, epoch, train_args):
	
	
	print(f"Running experiment with network '{network}' and dataset '{dataset}''")

	datasets_module = importlib.import_module("lab1.language_model.datasets.house_pred")
	dataset_class_ = getattr(datasets_module, dataset)
	# dataset_args = experiment_config.get("dataset_args", {})
	dataset = dataset_class_()
	# dataset.load_or_generate_data()

	models_module = importlib.import_module("lab1.language_model.models.base")
	model_class_ = getattr(models_module, model)

	networks_module = importlib.import_module("lab1.language_model.networks.mlp")
	network_fn = getattr(networks_module, network)
	# network_args = experiment_config.get("network_args", {})
	model = model_class_(dataset_cls=dataset_class_, network_fn=network_fn)
	

	train_model(
		model,
		dataset,
		epochs=epoch,
		# batch_size=experiment_config["train_args"]["batch_size"]
		)



if __name__ == "__main__":
	run_experiment()
	
