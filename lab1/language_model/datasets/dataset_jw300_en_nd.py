# dataset-jw300-en-nd.py
import os
from language_model.datasets.dataset import Dataset


class DatasetJW300(Dataset):
	""" 
	Class for datasets downloaded from the JW300 projecct
	"""
	
	# def load_or_generate(self):

		

	@classproperty
	def input_shape(self):
		""" returns the shape of self.data """
		pass

	@classproperty
	def text_to_token_sequences(self, token="word"):
		"""
		Converts self.data into sequences of tokens.
		returns self.data_sequence 
		"""
		pass

	def remove_duplicates(self):
		"""
		removes duplicates from self.data
		return self.data (almost) without duplicates 
		"""
		pass


	def normalize(self):
		""" Normalize self.data """
		
		













