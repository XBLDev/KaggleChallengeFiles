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


print("READING TRAIN AND TEST FILES...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("FINISHED READING TRAIN AND TEST FILES!")

##print(train.head())

print("PUTTING TRAIN AND TEST FILES TOGETHER...")
all_data = pd.concat((train.loc[:,'X0':'X385'],
                      test.loc[:,'X0':'X385']))
print("FINISHED PUTTING TRAIN AND TEST FILES TOGETHER!")



print("ANALYZING AND TRANSFORMING DATA FOR BETTER PERFORMANCE...")
#log transform the target:
##train["y"] = np.log1p(train["y"])

all_data = pd.get_dummies(all_data)
print("FINISHED ANALYZING AND TRANSFORMING DATA FOR BETTER PERFORMANCE!")
##
##
print("GETTING TRANSFORMED TRAINING DATA....")
X_train = all_data[:train.shape[0]]
##print(train.shape[0])##30471
print(X_train.shape)##(30471, 1886)
print("GOT TRANSFORMED TRAINING DATA")

print("GETTING TRANSFORMED TEST DATA....")
X_test = all_data[train.shape[0]:]
print(X_test.shape)##(7662, 1886)
print("GOT TRANSFORMED TEST DATA")
##
y = train.y

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)
params = {"max_depth":5, "eta":0.3}


xgb_params2 = {
    'eta': 0.05,##typical: 0.01 - 0.3
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.75,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'min_child_weight':75,
    'silent': 1,
    'seed':0,
    "gamma":0.5,
}

##xgbLinear uses: nrounds, lambda, alpha, eta
##xgbTree uses: nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight


LowestError = 10000
params = {"max_depth":3, "eta":1, "alpha": 1, "colsample_bytree":0.5, "subsample": 0.5 }
bestParams = params
bestBoostRound = 100
bestCVLength = 100000
##best_nrounds = int((res.shape[0] - estop) / (1 - 1 / n_folds))
##what's in cv's evluation history string; test-rmse-mean  test-rmse-std  train-rmse-mean  train-rmse-std
for currentMinChildWeight in np.arange(5, 105, 5):##1000, 10010, 10, got to roughly 660 on 8/06/2017, 7:10
    for currentEta in np.arange(0.01, 0.31, 0.1):
        for currentColsampleByTree in np.arange(0.5,1.05,0.05):
            for currentSubsample in np.arange(0.5,1.05,0.05):
                for currentGamma in np.arange(0,1.1,0.1):
                    for currentDep in np.arange(3, 16, 1):##3, 16, 1
                        currentParams = {"max_depth":currentDep, "gamma": currentGamma ,
                                         "eta":currentEta, "colsample_bytree":currentColsampleByTree,
                                         "subsample": currentSubsample, 'objective': 'reg:linear',
                                         'min_child_weight':currentMinChildWeight}
                        print("NEW CROSS VALIDATION RUNNING....")
                        print("currentEta: ", currentEta)
                        print("currentMinChildWeight: ", currentMinChildWeight)
                        print("currentColsampleByTree: ", currentColsampleByTree)
                        print("currentSubsample: ", currentSubsample)
                        print("currentGamma: ", currentGamma)
                        print("currentDep: ", currentDep)
                        print("currentParams: ",currentParams)

                        model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
                                       early_stopping_rounds=20, nfold = 5, verbose_eval=5)
                        

                        if model.values[model.shape[0]-1][0] < LowestError:
                            LowestError = model.values[model.shape[0]-1][0]
                            bestParams = currentParams
                            bestCVLength = model.shape[0]
                        print("LowestError: ", LowestError)
                        print("Best Parameter: ", bestParams )
                        print("Best CVLength: ", bestCVLength )
                        print(model)##[model.shape[0]-1]
                        f = open('MecedesPredictionXGB14_16_17.txt', 'w')
                        f.write("Best so far: ")
                        f.write('\n')
                        f.write('\n')
                        f.write(str(LowestError))
                        f.write('\n')
                        f.write(str(bestParams))
                        f.write('\n')
                        f.write(str(bestCVLength))
                        f.write('\n')
                        f.write('Possible boot round transformed:')
                        f.write(str(int((model.shape[0] - 20) / (1 - 1 / 5))))
                        f.write('\n')
                        f.write('\n')
                        f.write("Last tested set of parameters and BoostRound: ")
                        f.write('\n')
                        f.write('\n')
                        f.write(str(currentParams))
                        f.write('\n')
                        f.write(str(model.shape[0]))                        
                        f.close()
                        print("Written to file, go to next iteration")
                        print('\n')

##for currentBoostRound in np.arange(500, 50100, 100):##1000, 10010, 10, got to roughly 660 on 8/06/2017, 7:10
##    for currentEta in np.arange(0.01, 0.31, 0.1):
##        for currentColsampleByTree in np.arange(0.4,1.0,0.1):##0.5,1.0,0.1
##            for currentSubsample in np.arange(0.4,1.0,0.1):##0.5,1.0,0.1
##                for currentGamma in np.arange(0.1,10,0.1):##0.1,1.1,0.1
##                    for currentDep in np.arange(3, 16, 1):##3, 16, 1
##                        currentParams = {"max_depth":currentDep, "gamma": currentGamma ,"eta":currentEta, "alpha": 1, "colsample_bytree":currentColsampleByTree, "subsample": currentSubsample }
##                        print("NEW CROSS VALIDATION RUNNING....")
##                        print("currentBoostRound: ", currentBoostRound)
##                        print("currentEta: ",currentEta)
##                        print("currentColsampleByTree: ", currentColsampleByTree)
##                        print("currentSubsample: ", currentSubsample)
##                        print("currentGamma: ", currentGamma)
##                        print("currentDep: ", currentDep)
##                        print("currentParams: ",currentParams)
##
##                        model = xgb.cv(currentParams, dtrain,  num_boost_round=currentBoostRound, early_stopping_rounds=50, nfold = 5)
##                        
##
##                        if model.values[model.shape[0]-1][0] < LowestError:
##                            LowestError = model.values[model.shape[0]-1][0]
##                            bestBoostRound = currentBoostRound
##                            bestParams = currentParams
##                            bestCVLength = model.shape[0]
##                        print("LowestError: ", LowestError)
##                        print("Best BoostRound: ", bestBoostRound)
##                        print("Best Parameter: ", bestParams )
##                        print("Best CVLength: ", bestCVLength )
##                       
##                        f = open('MercedesPredictionCVRecord.txt', 'w')
##                        f.write("Best so far: ")
##                        f.write('\n')
##                        f.write('\n')
##                        f.write(str(LowestError))
##                        f.write('\n')
##                        f.write(str(bestBoostRound))
##                        f.write('\n')
##                        f.write(str(bestParams))
##                        f.write('\n')
##                        f.write(str(bestCVLength))
##                        f.write('\n')
##                        f.write('\n')
##                        f.write("Last tested set of parameters and BoostRound: ")
##                        f.write('\n')
##                        f.write('\n')
##                        f.write(str(currentBoostRound))
##                        f.write('\n')
##                        f.write(str(currentParams))
##                        f.write('\n')
##                        f.write(str(model.shape[0]))                        
##                        f.close()
##                        print("Written to file, go to next iteration")
##                        print('\n')












##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=1000,
##    max_depth=3,
##    learning_rate=0.01,
##    subsample=0.5,
##    colsample_bytree=0.5,
##    gamma=0.0,
##    ) #the params were tuned using xgb.cv
##
##model_xgb.fit(X_train, y)
##xgb_preds = np.expm1(model_xgb.predict(X_test))
##xgb_preds = np.around(xgb_preds, decimals=1)
##preds = xgb_preds
##
##solution = pd.DataFrame({"ID":test.ID, "y":preds})
##solution.to_csv("MercedesPredict.csv", index = False)
print("END OF PROGRAM")
