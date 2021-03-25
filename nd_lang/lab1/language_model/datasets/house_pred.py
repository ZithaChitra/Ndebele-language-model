from lab1.language_model.datasets.dataset import Dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler # Used for scaling of data
from sklearn.model_selection import train_test_split
# from keras.wrappers.scikit_learn import KerasRegressor


class HousingData(Dataset):
	pass

if __name__ == "__main__":
	# Read in train data
	df_train = pd.read_csv('train.csv', index_col=0)
	df_train.head()














