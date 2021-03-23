# dataset-jw300-en-nd.py
import os
from language_model.datasets.dataset import Dataset
from string import punctuation 
import nltk
from collections import Counter

nltk.download("punkt")
nltk.download("stopwords")


class DatasetJW300(Dataset):
	""" 
	Class for datasets downloaded from the JW300 projecct
	"""
	
	# def load_or_generate(self):
	self.sentences = [s for s in data.split("\n")]
	
	

	@classproperty
	def input_shape(self):
		""" returns the shape of self.data """
		print("Number of sentences: ", str(len(sentences)))


	
	def word_tokenizer(self, token="word"):
		"""
		Converts self.data into sequences of tokens.
		returns self.data_sequence 
		"""
		word_tokens = nltk.word_tokenize(sent_token)
		return word_tokens
		


	def remove_tokens(tokens: list, remove_tokens):
		"""
		Removes a token from a list of tokens.
		"""
		with_tokens_removed = []
		for token in tokens:
			if token not in remove_tokens:
				with_tokens_removed.append(token)
		return with_tokens_removed



	def lowercase(tokens: list):
		"""
		Converts string tokens into lowercase
		"""
		lower_case = []
		for token in tokens:
			lower_case.append(token.lower())
		return lower_case



	def remove_word_fragments(tokens: list):
		# fragments are: n't, 's
		tokens_no_frags = []
		for token in tokens:
			if "'" not in token:
				tokens_no_frags.append(token)
		return tokens_no_frags



	def remove_digits():
		for token in tokens:
			if token.isnumeric():
				tokens.remove(token)
		return tokens

	def preprocess_data(tokens: list):




#------------------------------------------------------------


	def remove_duplicates(self):
		"""
		removes duplicates from self.data
		return self.data (almost) without duplicates 
		"""
		pass


	def normalize(self):
		""" Normalize self.data """
		
		













