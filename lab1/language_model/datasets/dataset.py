# dataset.py
""" Base Dataset class to be extended by dataset-specific classes """

from pathlib import Path
import argparse
import os

# from language_model import util

class Dataset:
	"""
	Abstract class for datasets.
	A Dataset object will download it's relevant data if
	it is not already available on local storage.
	
	"""

	def __init__(self):
		self.data = self.load_or_generate()
		

	@classmethod
	def data_dirname(self):
		data_path = Path(__file__).resolve().parents[2] / "data"
		if not data_path.exists():
			print("Path does not exist")
		else:
			print("Path exists")
		return
		 

	def load_or_generate(self):
		"""
		Loads data if it is already in local storage, otherwise
		it dowloads it from external server.
		"""
		pass

	
	@property
	def input_shape(self):
		"""
		Returns the shape of the data variable
		"""
		pass

	@classmethod
	def preprocess_data(self):
		"""
		A combination of preprocessing steps that use different functions
		to achieve some transformation on the data.
		Example, using textual data:
		data = data.word_tokenizer()
		data = data.lowercase()
		data = data.romove_frags()
		e.t.c
		return data
		"""
		pass

	
	# def _download_raw_dataset(metadata):
	# 	if os.path.exists(metadata["filename"]):
	# 		return

	# 	print(f"Downloading raw dataset from {metadata[url]}...")
	# 	util.download_url(metadata["url"], metadata["filename"])
	# 	sha256 = util.compute_sha256(metadata["filename"])
	# 	if sha256 != metadata["sha256"]:
	# 		raise ValueError("Downloaded data file SHA-256 does not match that listed in the metadata document")


	def _parse_args(metadata):
		parser = argparse.ArgumentPaprse()
		parser.add_argument(
			"--subsample_fraction", type=float, default=None, help="If given, is used as fraction of data to expose."
			)
		return parser.parse_args()





	




















