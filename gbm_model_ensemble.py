# coding=gb2312
"""
blends the model 22043 and lgb_basic  
"""
import pandas as pd 

test = pd.read_csv("../dataset/test.csv", parse_dates=[3])

lgb_basic = pd.read_csv('output/test_prediction/lightgbm_test.csv')
lgb_22043 = pd.read_csv('output/test_prediction/test_22043.csv')

test_21026 = pd.read_csv('output/test_prediction/test_21026.csv')
test_20009 = pd.read_csv('output/test_prediction/test_20009.csv')


blending = (lgb_basic.Sales + lgb_22043.Sales + test_21026.Sales + test_20009.Sales) * 0.995 / 4

# output
result = pd.DataFrame({'Id': test['Id'], 'Sales': blending})
result.to_csv('output/test_prediction/ensemble_b_22043_21026_20009.csv', index=False)
print('Done')