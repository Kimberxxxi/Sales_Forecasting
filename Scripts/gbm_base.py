# coding=gb2312
"""
this script runs gbm baseline

"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess
from extract_features import extract_features
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

# 确保目录存在
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

def rmspe_lgb(yhat, y):
    y = np.expm1(y)  # Convert the labels back to the original scale
    yhat = np.expm1(yhat)  # Convert the predictions back to the original scale
    return "rmspe", rmspe(y, yhat)

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

# create feature map
def create_feature_map(features):
    outfile = open('lgbm.fmap', 'w')
    for i, feature in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feature))
    outfile.close()

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

# Convert to Dataset format
train_data = lgb.Dataset(X_train[features], label=y_train)
valid_data = lgb.Dataset(X_valid[features], label=y_valid)

# Train the model
print("Train a LightGBM model")
bst = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[train_data, valid_data],  # watchlist should contain both training and validation sets
    callbacks=callbacks
)

# Make predictions on the validation set
print('Performing validation')
yhat_valid = bst.predict(X_valid[features])

valid_result = pd.DataFrame({'Sales': np.expm1(yhat_valid)})
valid_result.to_csv('output/valid_prediction/valid_lightgbm.csv', index=False)
error = rmspe(np.expm1(y_valid), np.expm1(yhat_valid))
print('Validation RMSPE: {:.6f}'.format(error))

# Make predictions on the test set
print('Make predictions on the test set')
test_probs = bst.predict(test_df[features])
result = pd.DataFrame({'Id': test['Id'], 'Sales': np.expm1(test_probs)})
result.to_csv('output/test_prediction/lightgbm_test.csv', index=False)



print('Create feature map to get feature importance')
create_feature_map(features)

# Extract feature importance
importance = bst.feature_importance(importance_type='split')  # or use 'gain' for gain-based importance
importance = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)

# Create a DataFrame
df = pd.DataFrame(importance, columns=['feature', 'fscore'])

# Normalize to make it relative
df['fscore'] = df['fscore'] / df['fscore'].sum()

# Ensure the DataFrame is sorted in descending order by 'fscore'
df = df.sort_values(by='fscore', ascending=False)

# Save to CSV
df.to_csv('output/feature_importance_lightgbm.csv', index=False)

# Plot feature importance
featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('LightGBM Feature Importance')
plt.xlabel('Relative Importance')

# Save the plot
fig_featp = featp.get_figure()
fig_featp.savefig('output/feature_importance_lightgbm.png', bbox_inches='tight', pad_inches=1)