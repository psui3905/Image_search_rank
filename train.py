import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from spacy.lemmatizer import Lemmatizer 

import config

# Score: 
#   * n = predited_clicking_time
# 	* m = total_user_search_for that query
# 	score = n / m

# Features: 
# 	* f1 to f33
#	* 


def Data_Preprocessor(path:str):

	data = pd.read_csv("../{}.csv".format(path))
	query = data['query']
	data[['f1', 'f15']] = data[['f1', 'f15']].replace([True, False], [1,0])
	data.iloc[:,5:11] = data.iloc[:,5:11].replace([True, False], [1,0])

	print("Normolizing the Features...")
	df = data.iloc[:, 4:35]
	names = df.columns
	scaler = preprocessing.StandardScaler()
	scaled_df = scaler.fit_transform(df)
	data.iloc[:, 4:35] = pd.DataFrame(scaled_df, columns=names)
	data_collection = [data.iloc[:, 4:35].values, data.iloc[:, 35].values, data['media_id'].values, data['query'].values]
	print("Featuer dimension: {}".format(data_collection[0].shape))
	print("Y: {}".format(data_collection[1]))
	return data_collection

def train_cv_test_split(X, Y, train_rate, cv_rate, test_rate):
	m = X.shape[1]
    train = round(m*train_rate)
    cv = train + round(m*cv_rate)
    test = cv + round(m*test_rate)

    train_X = X[:,:train]
    train_Y = Y[:,:train]
    if(train_rate == 0):
        train_X = None

    cv_X = X[:,train:cv]
    cv_Y = Y[:, train:cv]
    if(cv_rate == 0):
        cv_X = None

    test_X = X[:, cv:]
    test_Y = Y[:, cv:]
    if(test_rate == 0):
        test_X = None

    return (train_X, train_Y, cv_X, cv_Y, test_X, test_Y)

    

def train_and_predit():

	(X, Y, media_id, query) = Data_Preprocessor("data")





