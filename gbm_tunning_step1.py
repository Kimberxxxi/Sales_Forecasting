# coding=gb2312
"""
tuuning Step 1：n_estimators

"""
import pandas as pd
import lightgbm as lgb
import numpy as np
from preprocess import preprocess
from extract_features import extract_features

# Load the data
print("Loading train, test, and store data...")
train = pd.read_csv("../dataset/train.csv", parse_dates=[2])
test = pd.read_csv("../dataset/test.csv", parse_dates=[3])
store = pd.read_csv("../dataset/store.csv")

# Preprocess the data
print("Preprocessing the data...")
preprocessed_df = preprocess(train, test, store)

# Extract features from the preprocessed data
print("Extracting features...")
features_df = extract_features([], preprocessed_df)

# Define RMSPE evaluation function
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))

# LightGBM evaluation function for RMSPE
def rmspe_lgb(yhat, data):
    y = np.expm1(data)  # Inverse transform the labels
    yhat = np.expm1(yhat)  # Inverse transform the predictions
    return "rmspe", rmspe(y, yhat), False  # Return RMSPE value

# Split train and test sets
train_df = features_df[features_df['Set'] == 1]
test_df = features_df[features_df['Set'] == 0]

# Create validation set based on the last 6 weeks of training data
timeDelta = test_df.Date.max() - test_df.Date.min()
maxDate = train_df.Date.max()
minDate = maxDate - timeDelta
valid_indices = train_df['Date'].apply(lambda x: minDate <= x <= maxDate)
train_indices = ~valid_indices

X_train = train_df[train_indices]
X_valid = train_df[valid_indices]
y_train = train_df['LogSales'][train_indices]
y_valid = train_df['LogSales'][valid_indices]

# Extract features used for the model
features_used = []
for model_index in [22043]:
    with open(f'output/features/features_{model_index}.txt', 'r') as ft:
        fts = ft.readlines()[0].split("'")
        features_used += [f for f in fts if len(f) > 2]

# Remove duplicates from features_used
features_used = list(set(features_used))

# Prepare the dataset for LightGBM
train_data = lgb.Dataset(X_train[features_used], label=y_train)
valid_data = lgb.Dataset(X_valid[features_used], label=y_valid)

# Set LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'seed': 42
}

# Cross-validation with LightGBM
cv_results = lgb.cv(
    params, train_data, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, 
    metrics='rmse', early_stopping=50, verbose_eval=50, show_stdv=True, seed=0
)

# Print the best number of estimators and the best CV score
print('Best n_estimators:', len(cv_results['rmse-mean']))
print('Best CV score:', cv_results['rmse-mean'][-1])

