# -------------------------------------------------------------
# _____Basics of text analysis (On a single document)_____
# 
#-> MEASUREMENTS
#
# 1. Basic Counts
# 2. Word Cloud
#
#++++++++++++++++++++++++
#
#-> Clean-Up
#
# 1. Remove punctuation 
# 2. Remove stop words
# 3. Normalize the case
# 4. Remove fragments
# 5. Stemming
#
#++++++++++++++++++++++++
#
#-> Text structure
#
# 1.Pos-taging
# 2.
# 3. 
#
#++++++++++++++++++++++++
#
#-> Word Relationships
#
# 1. Concordance
# 2. N-Grams ( co-location )
# 3. Co-occurence




# -------------------------------------------------------------
# _____Collections of documents (multiple documents)_____
#
#-> Term Significance
#
# 1. TF*IDF (Term frequency * Inverse Document frequency ) 
# a) TF = (Number of times a term appears in a document) / (Total number of terms in a document)
# b) IDF = log ( (Total number of documents) / (Number of documents with term t in it) )
# - A high weight in TF.IDF is reached by a high term frequency (in a given document) and
#   a low frequency in the number of documents that contain that term 
#
#++++++++++++++++++++++++
#
#-> Comparisons of documents 
#
# 1. Clustering
# 2. Cosine similarity
#
#++++++++++++++++++++++++
#_____And many more not included here...___

import subprocess
import nltk
from collections import Counter
nltk.download('punkt')
nltk.download('stopwords')




with open("example_file.txt", "r") as f:
	text = f.read()
	tokens = nltk.word_tokenize(text)
counts = Counter(tokens)
sorted_counts = sorted(counts.items(), key=lambda count: count[1], reverse=True)
print("len word_tokens", str(len(tokens)))


sentence_tokens = nltk.sent_tokenize(text)
counts = Counter(sentence_tokens)
sorted_counts = sorted(counts.items(), key=lambda count: count[1], reverse=True)
print("len sentence_tokens: ", str(len(sentence_tokens)))

# _____Romove punctuation_____
# punctuation is all none word and number characters except space
from string import punctuation

def remove_tokens(token, remove_tokens):
	return [token for token in tokens if token not in remove_tokens]

no_punc_tokens = remove_tokens(tokens, punctuation)
print("len no_punc_tokens: ", str(len(no_punc_tokens)))


#_____Normalize case_____
# lowercase every token
def lowercase(tokens):
	return [token.lower() for token in tokens]

lower_case_tokens = lowercase(no_punc_tokens)

#_____Remove stop words_____
# import stopwords from nltk
from nltk.corpus import stopwords
stops = stopwords.words("english")

no_stops_tokens = remove_tokens(lower_case_tokens, stops)
print("len no_stops_tokens", str(len(no_stops_tokens)))

#_____Remove Fragments_____
# remove fragmented words like:  n't, 's
def remove_word_fragments(tokens):
	return [token for token in tokens if "'" not in token]

no_frag_tokens = remove_word_fragments(no_stops_tokens)
print("len no_frag_tokens", str(len(no_frag_tokens)))


#_____Stemming_____
# converts words to their base form
# They don't always make sense and will sometimes lose their semantic meaning
from nltk.stem import PortStemmer
stemmer = PortStemmer()

stemmed_tokens = [stemmer.stem(token) for token in no_frag_tokens]




