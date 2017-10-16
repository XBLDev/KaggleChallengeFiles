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
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout


from keras import optimizers




print("READING TRAIN AND TEST FILES...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("FINISHED READING TRAIN AND TEST FILES!")

##print(train.head())

print("PUTTING TRAIN AND TEST FILES TOGETHER...")
all_data = pd.concat((train.loc[:,'timestamp':'market_count_5000'],
                      test.loc[:,'timestamp':'market_count_5000']))
print("FINISHED PUTTING TRAIN AND TEST FILES TOGETHER!")

##print(type(all_data))##<class 'pandas.core.frame.DataFrame'>
##print(all_data.head())

##matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)##12.0, 6.0
##prices = pd.DataFrame({"price_doc":train["price_doc"], "log(price_doc + 1)":np.log1p(train["price_doc"])})
##prices.hist()
##plt.show()


print("ANALYZING AND TRANSFORMING DATA FOR BETTER PERFORMANCE...")
#log transform the target:
train["price_doc"] = np.log1p(train["price_doc"])
#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
non_numeric_feats = all_data.dtypes[all_data.dtypes == "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
##print(skewed_feats)
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
print("FINISHED ANALYZING AND TRANSFORMING DATA FOR BETTER PERFORMANCE!")


print("GETTING TRANSFORMED TRAINING DATA....")
X_train = all_data[:train.shape[0]]
##print(train.shape[0])##30471
##print(X_train.shape)##(30471, 1886)
print("GOT TRANSFORMED TRAINING DATA")

print("GETTING TRANSFORMED TEST DATA....")
X_test = all_data[train.shape[0]:]
##print(X_test.shape)##(7662, 1886)
print("GOT TRANSFORMED TEST DATA")

y = train.price_doc
##print(y)

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75, 100, 150, 200]
##cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
##cv_ridge = pd.Series(cv_ridge, index = alphas)
##print(type(cv_ridge))##<class 'pandas.core.series.Series'>
##cv_ridge.plot(title = "Validation - Just Do It")
##plt.xlabel("alpha")
##plt.ylabel("rmse")
##plt.show()
##print("RIDGE ERROR MIN: ", cv_ridge.min())##RIDGE ERROR MIN:  0.48850972753

##lassoalphas = [1, 0.1, 0.001, 0.0005]
##lassoalphas2 = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]
##model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
##model_lasso_graph = pd.Series(rmse_cv(model_lasso).mean(), index = lassoalphas2)
####print("LASSO ERROR MIN: ",rmse_cv(model_lasso).mean())
##print("LASSO ERROR MIN: ",model_lasso_graph.min())
##model_lasso_graph.plot(title = "Validation For Lasso Model")
##plt.xlabel("alpha")
##plt.ylabel("rmse")
##plt.show()
##model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
##print("LASSO ERROR MIN: ",rmse_cv(model_lasso).mean())


##coef = pd.Series(model_lasso.coef_, index = X_train.columns)
##print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

##X_train = StandardScaler().fit_transform(X_train)
dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

##********************A PRAMS LIST FROM STACKOVERFLOW
##param <- list(objective = "multi:softprob",
##      eval_metric = "mlogloss",
##      num_class = 12,
##      max_depth = 8,
##      eta = 0.05,
##      gamma = 0.01, 
##      subsample = 0.9,
##      colsample_bytree = 0.8, 
##      min_child_weight = 4,
##      max_delta_step = 1
##      )
##cv.nround = 1000
##cv.nfold = 5
##mdcv <- xgb.cv(data=dtrain, params = param, nthread=6, 
##                nfold=cv.nfold, nrounds=cv.nround,
##                verbose = T)
##********************A PRAMS LIST FROM STACKOVERFLOW

##params = {"max_depth":2, "eta":0.2, "alpha": 1, "colsample_bytree":0.75, "subsample": 0.75 }
##model = xgb.cv(params, dtrain,  num_boost_round=1000, early_stopping_rounds=8, nfold = 10, shuffle=true)
##print(model)

model_xgb = xgb.XGBRegressor(
    n_estimators=360,
    max_depth=6,
    learning_rate=0.2,reg_alpha=1,subsample=0.75, colsample_bytree=0.75 ) #the params were tuned using xgb.cv


model_xgb.fit(X_train, y)
##xgb_preds = np.expm1(model_xgb.predict(X_test))

xgb_preds = model_xgb.predict(X_test)

##print(xgb_preds)##np.ndarray
##lasso_preds = np.expm1(model_lasso.predict(X_test))
##ridge_preds = np.expm1(cv_ridge.predict(X_test))
##predictions = pd.DataFrame({"xgb":xgb_preds, "ridge":ridge_preds})
##predictions.plot(x = "xgb", y = "ridge", kind = "scatter")
##plt.show()


##preds = 0.7*lasso_preds + 0.3*xgb_preds
preds = xgb_preds
solution = pd.DataFrame({"id":test.id, "price_doc":preds})
solution.to_csv("RussianHousingPredict.csv", index = False)



print("END OF PROGRAM")
