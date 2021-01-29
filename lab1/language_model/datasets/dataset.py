# dataset.py
""" Base Dataset class to be extended by dataset-specific classes """

from pathlib import Path
import argparse
import os
import opustools_pkg

from language_model import util

class Dataset:
	"""
	Abstract class for datasets.
	A Dataset object will download it's relevant data if
	it is not already available on local storage.
	Key features:
	1. shape: important for building model architecture
	   parameters ahead of time.
	"""
	self.data = load_or_generate()
	self.shape = # Function to get input shape


	@classmethod
	def data_dirname(self):
		return Path(__Path__).resolve() / "data"

	def load_or_generate(self, source="nd", target="en"):
		""" returns self.data """
		# TODO: Set your source and target languages. Keep in mind, these traditionally use language codes as found here:
		# These will also become the suffix's of all vocab and corpus files used throughout
		source_language = "en"
		target_language = "nd" 
		
		


	# def _download_raw_dataset(metadata):
	# 	if os.path.exists(metadata["filename"]):
	# 		return

	# 	print(f"Downloading raw dataset from {metadata[url]}...")
	# 	util.download_url(metadata["url"], metadata["filename"])
	# 	sha256 = util.compute_sha256(metadata["filename"])
	# 	if sha256 != metadata["sha256"]:
	# 		raise ValueError("Downloaded data file SHA-256 does not match that listed in the metadata document")




	def _parse_args(metadata):
		parser = argparser.ArgumentPaprse()
		parser.add_argument(
			"--subsample_fraction", type=float, default=None, help="If given, is used as fraction of data to expose."
			)
		return parser.parse_args()
































