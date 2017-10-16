import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

import xgboost as xgb


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

all_data = pd.concat((train_df.loc[:,'X0':'X385'],
                      test_df.loc[:,'X0':'X385']))

all_data = pd.get_dummies(all_data)

train_X = all_data[:train_df.shape[0]]

train_y = train_df.y

test_X = all_data[train_df.shape[0]:]

dtrain = xgb.DMatrix(train_X, label = train_y)

earlyStopRounds = 50
currentParams = {"max_depth":10, "gamma": 15.0,
                 "eta":0.01, "colsample_bytree":0.9,
                 "subsample": 0.9, 'objective': 'reg:linear',
                 'min_child_weight':800}

currentParams = {"max_depth":10,
                 "eta":0.05, "colsample_bytree":0.7,
                 "subsample": 0.7, 'objective': 'reg:linear',
                 }
##xgb_params = {
##    'eta': 0.05,
##    'max_depth': 5,
##    'subsample': 0.7,
##    'colsample_bytree': 0.7,
##    'objective': 'reg:linear',
##    'eval_metric': 'rmse',
##    'silent': 1
##}
SEED = 0
print(currentParams)
model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
print("ERROR: ", model.values[model.shape[0]-1][0])
print("BEST ROUND: ", model.shape[0])


##*****************BELOW IS THE FINAL MODEL AFTER WE GET THE BEST SET OF PARAMETERS***************************
##print("BEGIN GENERATING OUTPUT FILE......")
##earlyStopRounds = 50
##currentParams = {"max_depth":10, "gamma": 15.0,
##                 "eta":0.01, "colsample_bytree":0.9,
##                 "subsample": 0.9, 'objective': 'reg:linear',
##                 'min_child_weight':800}
##SEED = 0
##
##
##best_num_boost_round = 6187
##best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))
####
##
##model_xgb = xgb.XGBRegressor(
##    objective= "reg:linear",
##    n_estimators= best_num_boost_round,
##    max_depth=10,
##    learning_rate=0.01,
##    subsample=0.9,
##    colsample_bytree=0.9,
##    gamma=15.0,
##    ) #the params were tuned using xgb.cv
####
##model_xgb.fit(train_X, train_y)
##xgb_preds = model_xgb.predict(test_X)
####xgb_preds = np.around(xgb_preds, decimals=1)
##preds = xgb_preds
####
##solution = pd.DataFrame({"ID":test_df.ID, "y":preds})
##solution.to_csv("MercedesPredict.csv", index = False)


print("END OF PROGRAM")
