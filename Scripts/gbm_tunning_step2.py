# coding=gb2312
"""
tuuning Step 2：max_depth; subsample; colsample_bytree

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

# Hyperopt parameter tuning
def objective(params):
    model = lgb.LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        learning_rate=0.05,
        n_estimators=1000,  # You can adjust this as needed
        seed=42,
        max_depth=int(params['max_depth']),
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree']
    )

    # Define the callbacks for early stopping and logging
    callbacks = [
        lgb.callback.early_stopping(stopping_rounds=100),
        lgb.callback.log_evaluation(period=10)  # Log every 10 iterations
    ]
    
    model.fit(X_train[features_used], y_train, eval_set=[(X_valid[features_used], y_valid)], 
              eval_metric=rmspe_lgb, callbacks=callbacks)

    yhat = model.predict(X_valid[features_used])
    error = rmspe(X_valid.Sales.values, np.expm1(yhat))

    return {'loss': error, 'status': STATUS_OK}

# Define the parameter search space
space = {
    'max_depth': hp.choice('max_depth', [8, 9, 10]),
    'subsample': hp.uniform('subsample', 0.7, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
}


def optimize_params():
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )
    
    print("Best hyperparameters found: ", best)
    return best

# Optimize parameters using hyperopt
best_params = optimize_params()

# Train final model with best params
final_model = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    learning_rate=0.05,
    n_estimators=1000,
    seed=42,
    max_depth=int(best_params['max_depth']),
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree']
)

# Final training with the optimized parameters
final_model.fit(X_train[features_used], y_train)

# Make predictions on the test set
test_probs = final_model.predict(test_df[features_used])

# Create a DataFrame for the results
result = pd.DataFrame({'Id': test['Id'], 'Sales': np.expm1(test_probs)})

# Save predictions
result.to_csv('output/test_prediction/gbm_22043_tunning1.csv', index=False)

