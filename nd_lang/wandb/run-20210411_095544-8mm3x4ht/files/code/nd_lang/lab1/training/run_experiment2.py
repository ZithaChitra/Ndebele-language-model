""" Script to run an experiment """
import importlib
# from typing import Dict
# import os
import click

# from lab1.training.util import train_model
from lab1.training.util import save_net_artifact, save_data_raw_artifact, save_data_processed_artifact
from lab1.training.util_yaml import yaml_loader
import wandb
from wandb.keras import WandbCallback
# import numpy as np


DEFAULT_TRAIN_ARGS = {"batch_size":64, "epochs":16}


@click.command()
@click.argument("config_yaml", type=click.Path(exists=True), default="yamls/experiments/default.yaml")
@click.option("--train-args", default=DEFAULT_TRAIN_ARGS)
def run_experiment(config_yaml, train_args):


	exp_config = yaml_loader(config_yaml)
    
	model = exp_config.get("model")

	network = exp_config.get("network")
	net_cl_name = network["name"]
	net_config = network["network_args"]
	

	dataset = exp_config.get("dataset")
	data_cl_name = dataset["name"]
	dataset_args = dataset["dataset_args"]

	proj_name = exp_config.get("project_name")

	
	print(f"Running experiment with network '{net_cl_name}' and dataset '{data_cl_name}''")
	datasets_module = importlib.import_module("lab1.language_model.datasets.house_pred")
	dataset_class_ = getattr(datasets_module, data_cl_name)
	

	models_module = importlib.import_module("lab1.language_model.models.base2")
	model_class_ = getattr(models_module, model)

	networks_module = importlib.import_module("lab1.language_model.networks.mlp")
	network_fn = getattr(networks_module, net_cl_name)
	
	
	model = model_class_(dataset_cls=dataset_class_, network_fn=network_fn, dataset_args=dataset_args, network_args=net_config)


	# mlflow.set_tracking_uri("sqlite:///mlruns.db")
	# input_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, 13), name="house_attribs")])
	# output_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, 1), name="predicted house price")])
	# signature = ModelSignature(inputs=input_schema, outputs=output_schema)
	# input_example = np.array([[1., 2.5, 3. , 1.7, 2.1, 1.3, .5, .75, .89, 1.9, 2.15, 2.2, .6]])
	# mlflow.pyfunc.save_model(path="my_model", python_model=model, signature=signature, input_example=input_example )


	save_net_artifact(project_name=proj_name, network_fn=network_fn)
	save_data_raw_artifact(project_name=proj_name, data_class=dataset_class_)
	save_data_processed_artifact(project_name=proj_name, data_class=dataset_class_)
	with wandb.init(project=proj_name, config=exp_config) as run:
		config = wandb.config
        
		model.fit(dataset=config.dataset, callbacks=[WandbCallback()])


	# model_ = train_model(
	# 		model,
	# 		dataset,
	# 		epoch
	# 	)




if __name__ == "__main__":
	run_experiment()
	

