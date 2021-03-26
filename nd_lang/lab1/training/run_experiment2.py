""" Script to run an experiment """
import importlib
from typing import Dict
# import os
import click

from lab1.training.util import train_model

DEFAULT_TRAIN_ARGS = {"batch_size":64, "epochs":16}



@click.command()
@click.argument("dataset")
@click.argument("network")
@click.argument("model")
@click.option("--epoch", default=10)
@click.option("--batch-size")
@click.option("--train-args", default=DEFAULT_TRAIN_ARGS)
def run_experiment(dataset, network, model, epoch, ):
	"""
	Run a training experiment.

	Parameters
	----------
	experiment_config (dict)
		of the form
		{
			"dataset": "EmnistLineDataset",
			"dataset_args": {
				"max_overlap": 0.4,
				"subsample_fraction": 0.2
			},
			"model": "LineModel",
			"network": "line_cnn_all_conv",
			"network_args": {
				"window_width": 14,
				"window_stride": 7
			},
			"train_args": {
				"batch_size": 128,
				"epoch": 10
			}
		}

		save_weights (bool)
			If True, will save the final model weights to a canonical loaction
		gpu_ind (int)
			specifies which gpu to use (or -1 for first available)

		use_wandb (bool)
			sync training run to wandb

	"""
	# print(f"Running experiment with config {experiment_config} on GPU {gpu_ind}")

	datasets_module = importlib.import_module("lab1.language_model.datasets.housing_pred")
	dataset_class_ = getattr(datasets_module, dataset)
	# dataset_args = experiment_config.get("dataset_args", {})
	dataset = dataset_class_()
	dataset.load_or_generate_data()

	models_module = importlib.import_module("lab1.language_model.models.base")
	model_class_ = getattr(models_module, model)

	networks_module = importlib.import_module("lab1.language_model.networks.mlp")
	network_fn = getattr(networks_module, network)
	network_args = experiment_config.get("network_args", {})
	model = model_class_(
		dataset_cls=dataset_class_, network_fn=network_fn, dataset_args=dataset_args, network_args=network_args,
		)
	

	train_model(
		model,
		dataset,
		epochs=epoch,
		# batch_size=experiment_config["train_args"]["batch_size"]
		)







if __name__ == "__main__":
	run_experiment()
	
