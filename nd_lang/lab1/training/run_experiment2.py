""" Script to run an experiment """
# import importlib
from typing import Dict
# import os
import click

# from lab1.training.util import train_model

DEFAULT_TRAIN_ARGS = {"batch_size":64, "epochs":16}

@click.command()
@click.argument("experiment-config")
@click.option("--save-weights")
@click.option("--gpu-ind")
@click.option("--use-wandb")
def run_experiment(experiment_config: Dict, save_weights: bool, gpu_ind: int, use_wandb: bool = False):
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
	experiment_config = experiment_config
	print(f"exp_config: {experiment_config}")




if __name__ == "__main__":
	run_experiment()
	
