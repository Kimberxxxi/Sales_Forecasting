# coding=gb2312
"""
this script runs 100 models on random selections of the features

"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess
from extract_features import extract_features
import os
import random

# 确保目录存在
os.makedirs('output/features', exist_ok=True)
os.makedirs('output/model_infos', exist_ok=True)
# load the data 
print("Load the train, test and store data")
train = pd.read_csv("../dataset/train.csv", parse_dates=[2])
test = pd.read_csv("../dataset/test.csv", parse_dates=[3])
store = pd.read_csv("../dataset/store.csv")


# preprocess the data
print("Preprocess the data")
preprocessed_df = preprocess(train, test, store)

# define a feature list to store feature names
features = []
# extract features from preprocessed data
print("Extract features")
features_df = extract_features(features, preprocessed_df)


# Evaluation calculation
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)

# save features list as a txt file
def save_features(features, i, k):
	fn = 'features_' + str(16+k) + '{0:0=3d}'.format(i) + '.txt'
	with open('output/features/{}'.format(fn), 'w') as outfile:
		outfile.write(str(features))


# split train and test set
train_df = features_df[features_df['Set'] == 1]
test_df = features_df[features_df['Set'] == 0]

# use the last 6 weeks of the train set as validation set
timeDelta = test_df.Date.max() - test_df.Date.min()
maxDate = train_df.Date.max()
minDate = maxDate - timeDelta
# valid_indices is a list of boolean values which are true when date is within the last 6 weeks of train_df
valid_indices = train_df['Date'].apply(lambda x: (x >= minDate and x <= maxDate))
# train_indices is list of boolean values to get the train set
train_indices = valid_indices.apply(lambda x: (not x))

# split the train and valid set
X_train = train_df[train_indices]
X_valid = train_df[valid_indices]
y_train = train_df['LogSales'][train_indices]
y_valid = train_df['LogSales'][valid_indices]


# Parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'seed': 42
}

# Callbacks for early stopping and logging
callbacks = [
    lgb.callback.early_stopping(stopping_rounds=100),
    lgb.callback.log_evaluation(period=100)
]



# the number of feature selections performed 
num_of_models = 100
basic_features = ['DayOfYear','DayOfMonth','CompetitionDistance','DayOfWeek','AvgSalesPerCustomer','AvgSales',
				  'CompetitionOpenInMonth','Promo','WeekOfYear','Year',
				  'AvgSalesPerDow', 'PromoOpenInMonth','AvgCustomers','Store','medianSalesPerDow','holidays_thisweek']
total_sample_features = [
				  'AvgCustsPerDow','medianCustsPerDow',
				  'medianCustomers','Month','IsPromoMonth','StoreType',
						 'SchoolHoliday','Assortment','Promo2','StateHoliday','holidays_lastweek', 
						 'holidays_nextweek']

# a dict to save 100 models' infos
model_dicts = dict()

for i in range(num_of_models):
	model_info = {}

	# pick k features from total sample features list and k starts from 4 and is added one every 20 iterations
	k = int(i/20) + 4
	sample_features = random.sample(total_sample_features, k)
	features_used = basic_features + sample_features
	model_info['features_used'] = features_used
	save_features(features_used, i, k)

	print("train No.{} xgboost model".format(i))
	# Convert to Dataset format
	train_data = lgb.Dataset(X_train[features_used], label=y_train)
	valid_data = lgb.Dataset(X_valid[features_used], label=y_valid)

	# Train the model
	bst = lgb.train(
		params,
		train_data,
		num_boost_round=5000,
		valid_sets=[train_data, valid_data],  # watchlist should contain both training and validation sets
		callbacks=callbacks
	)
	

	model_name = str(16+k) + '{0:0=3d}'.format(i)

	yhat_valid = bst.predict(X_valid[features_used])
	valid_result = pd.DataFrame({'Sales': np.expm1(yhat_valid)})
	valid_result.to_csv('output/valid_prediction/{}.csv'.format('valid_'+model_name), index=False)

	error = rmspe(np.expm1(y_valid), np.expm1(yhat_valid))
	model_info['valid_error'] = error

	model_dicts[model_name] = model_info
	#output
	print('Make predictions on the test set')
	test_probs = bst.predict(test_df[features_used])
	result = pd.DataFrame({'Id': test['Id'], 'Sales': np.expm1(test_probs)})
	result.to_csv('output/test_prediction/{}.csv'.format('test_'+model_name), index=False)



models_df = pd.DataFrame(model_dicts).T 
models_df.to_csv('output/model_infos.csv')
