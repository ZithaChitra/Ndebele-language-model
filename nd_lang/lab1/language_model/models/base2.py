""" Model class, to be extended by specific types of models. """
# pylint: disable=missing-function-docstring
from pathlib import Path
from typing import Callable, Dict, Optional

from tensorflow import keras
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import RMSprop
# import mlflow
# import pandas
import numpy as np
# import mlflow.pyfunc


DIRNAME = Path(__file__).parents[1].resolve() / "weights"

class Model():
    """ 
	Base class, to be subclassed by predictors for specific types of data.
	This is a wrapper that makes  it convinient to use different neural net
	configurations during experiments. Configurations could be a different
	neural net archicture, dataset or maybe just hyperparameters.
	
	Parameters:
	----------
	dataset_cls: type
		Name of class that interfaces with your dataset.
	network_fn: Callable[..., KerasModel]
		Name of function that returns the KerasModel to be used for training.
	dataset_args: Dict
		A dictionary of arguments for modifying the dataset
	network_args:
		A dictionary of arguments for creating model
	"""
    def __init__(
        self,
        dataset_cls: type,
        network_fn: Callable[..., KerasModel],
        dataset_args: Dict = None,
        network_args: Dict = None,        
    ):
        self.name = f"{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}"

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(network_args)
        self.network.summary()

        self.batch_argument_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    @property
    def image_shape(self):
        return self.data.input_shape


    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f"{self.name}_weights.h5")

    def fit(
        self, dataset, batch_size: int = 32,
        epochs: int = 10, augment_val: bool = True, 
        callbacks: list = None
    ):
        if callbacks is None:
            callbacks = []
        
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

        # train_sequence = Dataset(
        #     dataset.x_train,
        #     dataset.y_train,
        #     batch_size,
        #     augment_fn=self.batch_augment_fn,
        #     format_fn=self.batch_format_fn,
        # )

        # test_sequence = Dataset(
        #     dataset.x_test,
        #     dataset.y_test,
        #     batch_size,
        #     augment_fn=self.batch_augment_fn if augment_val else None,
        #     format_fn=self.batch_format_fn
        # )

        self.network.fit(
            self.data.X_tr, self.data.y_tr,
            epochs=epochs,
            callbacks=callbacks,
            use_multiprocessing=False,
            workers=1,
            shuffle=True,
        )


    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """
		Function for making predictions and scoring the model
		"""
        self.network.predict(model_input)

    # def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 16, verbose: bool = True):
    #     # pylint: disable=unused-argument
    #     sequence = Dataset(x, y, batch_size=batch_size)
    #     preds = self.network.predict(sequence)
    #     return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))

    def loss(self):
        return "mse"


    def optimizer(self):
        optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")
        return optimizer

    def metrics(self):
        return ["accuracy"]

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)









		