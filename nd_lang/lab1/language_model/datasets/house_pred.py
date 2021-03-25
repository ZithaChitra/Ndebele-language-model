from lab1.language_model.datasets.dataset import Dataset
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HousingData(Dataset):
	def __init__(self):
		self.X, self.y = load_boston(return_X_y=True)


	def preprocess_data(self):
		std_scaler = StandardScaler()
		self.X_tr = std_scaler.fit_transform(self.X)
		self.y_tr = std_scaler.fit_transform(self.y.reshape(-1, 1))
		self.input_shape = self.X_tr.shape
		self.output_shape = self.y_tr.shape
		return self.X_tr, self.y_tr

	
	# def split_data(self, ratio: float=0.2):
	# 	X, y = self.preprocess_data()
	# 	self.X_train, self.X_test = train_test_split(X, ratio, shuffle=False)
	# 	self.y_train, self.y_test = train_test_split(y, ratio, shuffle=False)
		# return (self.X_train, self.y_train), (self.X_test, self.y_test)



	def data_info(self):
		print(f"X.shape: {self.X.shape}\ny.shape: {self.y.shape}")
		print(f"X.shape: {self.X_tr.shape}\ny.shape: {self.y_tr.shape}")
		# print(f"X.shape: {self.X_train.shape}\ny.shape: {self.y_train.shape}")
		# print(f"X.shape: {self.X_test.shape}\ny.shape: {self.y_test.shape}")




if __name__ == "__main__":
	housing_data = HousingData()
	housing_data.preprocess_data()
	# housing_data.split_data(ratio=0.2)
	housing_data.data_info()













