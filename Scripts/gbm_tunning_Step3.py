# coding=gb2312
"""
tuuning Step 3：learning_rate; num_boost_round; early_stopping_rounds

"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess
from extract_features import extract_features
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error

# 确保目录存在
os.makedirs('output/valid_prediction', exist_ok=True)
os.makedirs('output/valid_prediction', exist_ok=True)

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

# Evaluation function for RMSPE
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))

# LightGBM's evaluation function for RMSPE
# LightGBM's evaluation function for RMSPE
def rmspe_lgb(yhat, data):
    y = np.expm1(data)  # Directly use data as labels (they are already in numpy.ndarray)
    yhat = np.expm1(yhat)  # Convert the predictions back to the original scale
    return "rmspe", rmspe(y, yhat), False  # "rmspe" as the metric name, rmspe value, False (lower is better)


# split train and test set
train_df = features_df[features_df['Set'] == 1]
test_df = features_df[features_df['Set'] == 0]

# use the last 6 weeks of the train set as validation set
timeDelta = test_df.Date.max() - test_df.Date.min()
maxDate = train_df.Date.max()
minDate = maxDate - timeDelta
valid_indices = train_df['Date'].apply(lambda x: (x >= minDate and x <= maxDate))
train_indices = valid_indices.apply(lambda x: (not x))

X_train = train_df[train_indices]
X_valid = train_df[valid_indices]
y_train = train_df['LogSales'][train_indices]
y_valid = train_df['LogSales'][valid_indices]

# merge the features from 5 models above in the model description
features_used = []
for model_index in [22043]:
    with open('output/features/{}.txt'.format('features_' + str(model_index)), 'r') as ft:
        fts = ft.readlines()[0].split("'")
        fts = [f for f in fts if len(f) > 2]
        features_used += fts

features_used = list(set(features_used))




params = {
    'objective': 'regression',
    'metric': 'rmse',
    'eta': 0.03,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'max_depth': 10,
    'seed': 42
}

# Callbacks for early stopping and logging
callbacks = [
    lgb.callback.early_stopping(stopping_rounds=500),
    lgb.callback.log_evaluation(period=100)
]

# Convert to Dataset format
train_data = lgb.Dataset(X_train[features_used], label=y_train)
valid_data = lgb.Dataset(X_valid[features_used], label=y_valid)

# Train the model
print("Train a LightGBM model")
bst = lgb.train(
    params,
    train_data,
    num_boost_round=10000,
    valid_sets=[train_data, valid_data],  # watchlist should contain both training and validation sets
    callbacks=callbacks
)

# Make predictions on the validation set
print('Performing validation')
yhat_valid = bst.predict(X_valid[features_used])

valid_result = pd.DataFrame({'Sales': np.expm1(yhat_valid)})
valid_result.to_csv('output/valid_prediction/valid_22043_tunning3.csv', index=False)
error = rmspe(np.expm1(y_valid), np.expm1(yhat_valid))
print('Validation RMSPE: {:.6f}'.format(error))

# Make predictions on the test set
print('Make predictions on the test set')
test_probs = bst.predict(test_df[features_used])
result = pd.DataFrame({'Id': test['Id'], 'Sales': np.expm1(test_probs)})
result.to_csv('output/test_prediction/gbm_22043_tunning3.csv', index=False)


