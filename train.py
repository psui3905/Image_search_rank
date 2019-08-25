import pandas as pd 
import numpy as np 
from spacy.lemmatizer import Lemmatizer 

import config
import tensorflow as tf
from tensorflow import keras
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.utils import shuffle


@dataclass
class regression_set:
	media_id: list
	X: list
	Y: list

class Linear_regression_model:

	def __init__(self, X, Y, c_X, c_Y, media_id):
		self.X = X
		self.Y = Y
		self.c_X = c_X
		self.c_Y = c_Y
		self.media_id = media_id
		self.model = tf.keras.models.Sequential([
						keras.layers.Flatten(input_shape=(X.shape[1], 1)),
						keras.layers.Dense(256,activation=tf.nn.relu),
						keras.layers.Dropout(rate = 0.3),
						keras.layers.Dense(128,activation=tf.nn.relu),
						keras.layers.Dropout(rate = 0.3),
						keras.layers.Dense(1,activation=tf.nn.sigmoid)
						])

	def train(self, loss_func='mean_squared_error', optimizer='RMSprop', epochs=15):
		if optimizer == 'RMSprop':
			self.model.compile(optimizer=keras.optimizers.RMSprop(0.001),
						loss=loss_func,
						metrics=['mean_absolute_error', 'mean_squared_error'])
		elif optimizer == 'adam':
			self.model.compile(optimizer='adam',
						loss=loss_func,
						metrics=['mean_absolute_error', 'mean_squared_error'])
		else:
			print("Error: Unknow Activaition function")
			return
		history = self.model.fit(self.X, self.Y, epochs=epochs)
		return history.epoch, history.history['mean_squared_error'][-1]

	def predit(self):
		result = self.model.predict(self.c_X)
		return (result, self.c_Y)


def Data_Preprocessor(path):

	data = pd.read_csv("{}.csv".format(path))
	query = data['query']
	data[['f1', 'f15']] = data[['f1', 'f15']].replace([True, False], [1,0])
	data.iloc[:,5:11] = data.iloc[:,5:11].replace([True, False], [1,0])

	print("Normolizing the Features...")

	df = data.iloc[:, 4:35]
	names = df.columns
	scaled_df = preprocessing.scale(df.values)
	data.iloc[:, 4:35] = pd.DataFrame(scaled_df, columns=names)
	data = data.drop(['f2'], axis=1)
	print(data.shape)

	regression_dic = {}
	# print(data['query'].unique())
	for unique_query in data['query'].unique():
		specific_query_data = data.loc[data['query'] == unique_query]

		media_id_list = list(specific_query_data['media_id'].values)
		
		feature = specific_query_data.iloc[:, 2:34]
		specific_query_x = feature.values
		specific_query_x = list(specific_query_x)

		y_column = specific_query_data.iloc[:,34]
		y_sum = y_column.sum()
		specific_query_y = (y_column / y_sum)
		specific_query_y = list(specific_query_y.values)

		regression_dic[unique_query] = regression_set(
			media_id=media_id_list,
			X=specific_query_x,
			Y=specific_query_y,
			)

	return regression_dic


def train_validation_split(input_X, input_Y, train_rate, validation_rate):
	
	# shuffle the data
	X, Y = shuffle(input_X, input_Y, random_state=0)

	m = X.shape[0]
	train = round(m*train_rate)
	v = train + round(m*validation_rate)

	train_X = X[:train,:]
	train_X = train_X[..., np.newaxis]
	train_Y = Y[:train,]
	train_Y = train_Y[..., np.newaxis]
	
	valication_X = X[train:v,:]
	valication_X = valication_X[..., np.newaxis]
	validation_Y = Y[train:v,]
	validation_Y = validation_Y[..., np.newaxis]

	return (train_X, train_Y, valication_X, validation_Y)

def train_and_predit():

	regression_dic = Data_Preprocessor("data")
	model_dic = {}
	evaluate_dic = {}

	for query, data_set in regression_dic.items():
		source_X = np.array(data_set.X)
		source_Y = np.array(data_set.Y)
		source_media_id = data_set.media_id
		(train_X, train_Y, valication_X, validation_Y) = train_validation_split(source_X, source_Y, config.Training_Rate, config.Cross_Validate_Rate)
		model_dic[query] = Linear_regression_model(train_X, train_Y, valication_X, validation_Y, source_media_id)

	with open("predictions.tsv", "w+") as file:
		for query, model in model_dic.items():
			model.train()
			(predit_y, truth_y) = model.predit()
			predit_y = list(predit_y)
			truth_y = list(truth_y)
			for index in range(len(predit_y)):
				file.write("{}\t{}\t{}\n".format(query,truth_y[index][0],predit_y[index][0]))
	return

if __name__ == "__main__":
	train_and_predit()


