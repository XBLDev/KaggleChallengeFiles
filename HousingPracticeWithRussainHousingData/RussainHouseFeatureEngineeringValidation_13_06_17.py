##xgbLinear uses: nrounds, lambda, alpha, eta
##xgbTree uses: nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn import model_selection, preprocessing
import xgboost as xgb
from xgboost import XGBClassifier 
from random import randint
import random
import sys
import xlrd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from numpy import sort
import operator



np.set_printoptions(threshold=sys.maxsize)
color = sns.color_palette()
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)

def feat_imp(df, model, n_features):

    d = dict(zip(df.columns, model.feature_importances))
    ss = sorted(d, key=d.get, reverse=True)
    top_names = ss[0:n_features]

    plt.figure(figsize=(150,150))
    plt.title("Feature importances")
    plt.bar(range(n_features), [d[i] for i in top_names], color="r", align="center")
    plt.xlim(-1, n_features)
    plt.xticks(range(n_features), top_names, rotation='vertical')
    plt.show()
 
# From here: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity/notebook
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]
macro_cols2 = [ 
  "mortgage_value", "mortgage_rate"]


##train_df = pd.read_csv("train_clean.csv", parse_dates=['timestamp'])##, index_col='id'
##test_df = pd.read_csv("test_clean.csv", parse_dates=['timestamp'])




print("FIXING TRAIN AND TEST DATA WITH BAD_ADDRESS_FIX.xlsx...")
##train_df = pd.read_csv("train_clean.csv", parse_dates=['timestamp'])##, index_col='id'
##print(train_df.ix[train_df['id'] == 930]['sub_area'])
train_df = pd.read_csv("train_clean.csv", parse_dates=['timestamp'], index_col='id')##, index_col='id'
test_df = pd.read_csv("test_clean.csv", parse_dates=['timestamp'], index_col='id')
macro_df = pd.read_csv("macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols2)
fix = pd.read_excel('BAD_ADDRESS_FIX.xlsx').drop_duplicates('id').set_index('id')
train_df.update(fix, overwrite=True)
test_df.update(fix, overwrite=True)
##print('Fix in train: ', train_df.index.intersection(fix.index).shape[0])
##print(train_df.index.intersection(fix.index))
##print('Fix in test: ', test_df.index.intersection(fix.index).shape[0])
##print(test_df.index.intersection(fix.index))
train_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)
##print(train_df.ix[train_df['id'] == 930]['sub_area'])
##print(train_df['id'])
##print(test_df['id'])
print("TRAIN AND TEST DATA FIXED WITH BAD_ADDRESS_FIX.xlsx!")



##**********************************************BELOW FIX DATA BY PREDICTING MISSING DATA WITH EXISTING ONES*********************************************###
##print("FIXING DATA BY PREDICTING MISSING DATA WITH EXISTING ONES...")
##
##all_data = pd.concat((train_df.loc[:,'timestamp':'market_count_5000'], test_df.loc[:,'timestamp':'market_count_5000']))
##
##train_df_without_NaN = train_df.dropna()
##test_df_without_NaN = test_df.dropna()
##all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
##                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))
##
##
##all_data_life_sq_NaN = all_data['life_sq'].ix[(all_data['life_sq'].isnull() == True)]
##all_data_life_sq_NaN_index = all_data['life_sq'].index[all_data['life_sq'].apply(np.isnan)]
##
##train_data_life_sq_NaN = train_df['life_sq'].ix[(train_df['life_sq'].isnull() == True)]
##train_data_life_sq_NaN_index = train_df['life_sq'].index[train_df['life_sq'].apply(np.isnan)]
##
##test_data_life_sq_NaN = test_df['life_sq'].ix[(test_df['life_sq'].isnull() == True)]
##test_data_life_sq_NaN_index = test_df['life_sq'].index[test_df['life_sq'].apply(np.isnan)]
##
##all_data_num_room_NaN = all_data['num_room'].ix[(all_data['num_room'].isnull() == True)]
##all_data_num_room_NaN_index = all_data['num_room'].index[all_data['num_room'].apply(np.isnan)]
##
##train_data_num_room_NaN = train_df['num_room'].ix[(train_df['num_room'].isnull() == True)]
##train_data_num_room_NaN_index = train_df['num_room'].index[train_df['num_room'].apply(np.isnan)]
##
##test_data_num_room_NaN = test_df['num_room'].ix[(test_df['num_room'].isnull() == True)]
##test_data_num_room_NaN_index = test_df['num_room'].index[test_df['num_room'].apply(np.isnan)]
##
##all_data_kitch_sq_NaN = all_data['kitch_sq'].ix[(all_data['kitch_sq'].isnull() == True)]
##all_data_kitch_sq_NaN_index = all_data['kitch_sq'].index[all_data['kitch_sq'].apply(np.isnan)]
##
##train_data_kitch_sq_NaN = train_df['kitch_sq'].ix[(train_df['kitch_sq'].isnull() == True)]
##train_data_kitch_sq_NaN_index = train_df['kitch_sq'].index[train_df['kitch_sq'].apply(np.isnan)]
##
##test_data_kitch_sq_NaN = test_df['kitch_sq'].ix[(test_df['kitch_sq'].isnull() == True)]
##test_data_kitch_sq_NaN_index = test_df['kitch_sq'].index[test_df['kitch_sq'].apply(np.isnan)]
##
##all_data_school_quota_NaN = all_data['school_quota'].ix[(all_data['school_quota'].isnull() == True)]
##all_data_school_quota_NaN_index = all_data['school_quota'].index[all_data['school_quota'].apply(np.isnan)]
##
##train_data_school_quota_NaN = train_df['school_quota'].ix[(train_df['school_quota'].isnull() == True)]
##train_data_school_quota_NaN_index = train_df['school_quota'].index[train_df['school_quota'].apply(np.isnan)]
##
##test_data_school_quota_NaN = test_df['school_quota'].ix[(test_df['school_quota'].isnull() == True)]
##test_data_school_quota_NaN_index = test_df['school_quota'].index[test_df['school_quota'].apply(np.isnan)]
##
##all_data_preschool_quota_NaN = all_data['preschool_quota'].ix[(all_data['preschool_quota'].isnull() == True)]
##all_data_preschool_quota_NaN_index = all_data['preschool_quota'].index[all_data['preschool_quota'].apply(np.isnan)]
##
##train_data_preschool_quota_NaN = train_df['preschool_quota'].ix[(train_df['preschool_quota'].isnull() == True)]
##train_data_preschool_quota_NaN_index = train_df['preschool_quota'].index[train_df['preschool_quota'].apply(np.isnan)]
##
##test_data_preschool_quota_NaN = test_df['preschool_quota'].ix[(test_df['preschool_quota'].isnull() == True)]
##test_data_preschool_quota_NaN_index = test_df['preschool_quota'].index[test_df['preschool_quota'].apply(np.isnan)]
##
##all_data_max_floor_NaN = all_data['max_floor'].ix[(all_data['max_floor'].isnull() == True)]
##all_data_max_floor_NaN_index = all_data['max_floor'].index[all_data['max_floor'].apply(np.isnan)]
##
##train_data_max_floor_NaN = train_df['max_floor'].ix[(train_df['max_floor'].isnull() == True)]
##train_data_max_floor_NaN_index = train_df['max_floor'].index[train_df['max_floor'].apply(np.isnan)]
##
##test_data_max_floor_NaN = test_df['max_floor'].ix[(test_df['max_floor'].isnull() == True)]
##test_data_max_floor_NaN_index = test_df['max_floor'].index[test_df['max_floor'].apply(np.isnan)]
##
##
##all_data_full_sq_NaN = all_data['full_sq'].ix[(all_data['full_sq'].isnull() == True)]
##all_data_full_sq_NaN_index = all_data['full_sq'].index[all_data['full_sq'].apply(np.isnan)]
##
##train_data_full_sq_NaN = train_df['full_sq'].ix[(train_df['full_sq'].isnull() == True)]
##train_data_full_sq_NaN_index = train_df['full_sq'].index[train_df['full_sq'].apply(np.isnan)]
##
##test_data_full_sq_NaN = test_df['full_sq'].ix[(test_df['full_sq'].isnull() == True)]
##test_data_full_sq_NaN_index = test_df['full_sq'].index[test_df['full_sq'].apply(np.isnan)]
##
##all_data_build_year_NaN = all_data['build_year'].ix[(all_data['build_year'].isnull() == True)]
##all_data_build_year_NaN_index = all_data['build_year'].index[all_data['build_year'].apply(np.isnan)]
##
##all_data_build_year_NotNaN = all_data['build_year'].ix[(all_data['build_year'].isnull() == False)]
##all_data_build_year_NotNaN_index = all_data['build_year'].index[(all_data['build_year'].isnull() == False)]
##
##train_data_build_year_NaN = train_df['build_year'].ix[(train_df['build_year'].isnull() == True)]
##train_data_build_year_NaN_index = train_df['build_year'].index[train_df['build_year'].apply(np.isnan)]
##
##test_data_build_year_NaN = test_df['build_year'].ix[(test_df['build_year'].isnull() == True)]
##test_data_build_year_NaN_index = test_df['build_year'].index[test_df['build_year'].apply(np.isnan)]
##
##all_data_floor_NaN = all_data['floor'].ix[(all_data['floor'].isnull() == True)]
##all_data_floor_NaN_index = all_data['floor'].index[all_data['floor'].apply(np.isnan)]
##
##train_data_floor_NaN = train_df['floor'].ix[(train_df['floor'].isnull() == True)]
##train_data_floor_NaN_index = train_df['floor'].index[train_df['floor'].apply(np.isnan)]
##
##test_data_floor_NaN = test_df['floor'].ix[(test_df['floor'].isnull() == True)]
##test_data_floor_NaN_index = test_df['floor'].index[test_df['floor'].apply(np.isnan)]
##
####***************************FIX floor: add median +- 1**********************************************************************************************************
##
##print("ADDING MISSING floor VALUES...")
##
##train_df['floor'].ix[(train_df['floor'].isnull() == True)] = train_df['floor'].ix[(train_df['floor'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['floor'].median()-1, all_data_without_NaN['floor'].median()+1))
##test_df['floor'].ix[(test_df['floor'].isnull() == True)] = test_df['floor'].ix[(test_df['floor'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['floor'].median()-1, all_data_without_NaN['floor'].median()+1))
##
##all_data['floor'].ix[(all_data['floor'].isnull() == True)] = all_data['floor'].ix[(all_data['floor'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['floor'].median()-1, all_data_without_NaN['floor'].median()+1))
##
##print("ADDED MISSING floor VALUES")
##
####***************************FIX build_year with: 'floor','bulvar_ring_km','zd_vokzaly_avto_km','sadovoe_km','kremlin_km','ttk_km'**********************************************************************************************************
##
##print("ADDING MISSING build_year VALUES...")
##
##train_df_without_NaN = train_df.dropna()
##test_df_without_NaN = test_df.dropna()
##all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
##                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))
##
##DataMostCorrelatedToBuildYear = all_data_without_NaN[['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km',
##                                                     'kremlin_km', 'ttk_km']]
##
##train_X = DataMostCorrelatedToBuildYear[['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km',
##                                                     'kremlin_km', 'ttk_km']].values
##
##train_y = all_data_without_NaN['build_year']
##
##test_X = all_data.ix[(all_data['build_year'].isnull() == False)][['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km', 'kremlin_km', 'ttk_km']].values
##
##build_year_test_X = all_data.ix[(all_data['build_year'].isnull() == False)][['build_year']].values
##
##dtrain = xgb.DMatrix(train_X, label = train_y)
##
##earlyStopRounds = 10
##currentParams = {"max_depth":4, "gamma": 0.1,
##                 "eta":0.01, "colsample_bytree":0.75,
##                 "subsample": 0.75, 'objective': 'reg:linear',
##                 'min_child_weight':1}
##SEED = 0
##
####model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
####        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
####print("ERROR: ", model.values[model.shape[0]-1][0])
####print("BEST ROUND: ", model.shape[0])
##
##best_num_boost_round = 4213
##best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))
##
##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=best_num_boost_round,
##    max_depth=4,
##    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
##    gamma=0.1)
##
##model_xgb.fit(train_X, train_y)
##
##y_pred = model_xgb.predict(test_X)
##y_pred = np.around(y_pred)
##
##meanDifference = np.mean(y_pred - build_year_test_X)
##
##BuildYearsToPredict_Train = train_df.ix[(train_df['build_year'].isnull() == True)][['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km', 'kremlin_km', 'ttk_km']].values
##PredictedMissingBuildYears_Train = model_xgb.predict(BuildYearsToPredict_Train)
##PredictedMissingBuildYears_Train = np.around(PredictedMissingBuildYears_Train - meanDifference)
##
##BuildYearsToPredict_Test = test_df.ix[(test_df['build_year'].isnull() == True)][['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km', 'kremlin_km', 'ttk_km']].values
##PredictedMissingBuildYears_Test = model_xgb.predict(BuildYearsToPredict_Test)
##PredictedMissingBuildYears_Test = np.around(PredictedMissingBuildYears_Test - meanDifference)
##
##
##
##mean_build_year_before_train = train_df['build_year'].mean()
##mean_build_year_before_test = test_df['build_year'].mean()
##
##countOfPredictedValue_train = 0
##for value in train_data_build_year_NaN_index:
##    train_df.iloc[value,7] = PredictedMissingBuildYears_Train[countOfPredictedValue_train]
##    countOfPredictedValue_train = countOfPredictedValue_train + 1
##
##countOfPredictedValue_test = 0
##for value in test_data_build_year_NaN_index:
##    test_df.iloc[value,7] = PredictedMissingBuildYears_Test[countOfPredictedValue_test]
##    countOfPredictedValue_test = countOfPredictedValue_test + 1
##
##mean_build_year_after_train = train_df['build_year'].mean()
##mean_build_year_after_test = test_df['build_year'].mean()
##
##DifferenceOfMean_train = mean_build_year_before_train - mean_build_year_after_train
##DifferenceOfMean_test = mean_build_year_before_test - mean_build_year_after_test
##
##print("ADDED MISSING build_year VALUES")
##
####***************************Fix max_floor with: floor, build_year**********************************************************************************************************
##
##print("ADDING MISSING max_floor VALUES...")
##
##train_df_without_NaN = train_df.dropna()
##test_df_without_NaN = test_df.dropna()
##all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
##                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))
##
##DataMostCorrelatedTo_Max_Floor = all_data_without_NaN[['floor', 'build_year']]
##
##train_X = DataMostCorrelatedTo_Max_Floor.values
##train_y = all_data_without_NaN['max_floor']
##test_X = all_data.ix[(all_data['max_floor'].isnull() == False)][['floor', 'build_year']].values
##max_floor_test_X = all_data.ix[(all_data['max_floor'].isnull() == False)][['max_floor']].values
##
##
##dtrain = xgb.DMatrix(train_X, label = train_y)
##
##earlyStopRounds = 10
##currentParams = {"max_depth":4, "gamma": 0.1,
##                 "eta":0.01, "colsample_bytree":0.75,
##                 "subsample": 0.75, 'objective': 'reg:linear',
##                 'min_child_weight':1}
##SEED = 0
##
####model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
####        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
####print("ERROR: ", model.values[model.shape[0]-1][0])
####print("BEST ROUND: ", model.shape[0])
##
####ERROR:  3.0662888
####BEST ROUND:  761
##best_num_boost_round = 761
##best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))
##
##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=best_num_boost_round,
##    max_depth=4,
##    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
##    gamma=0.1)
##
##model_xgb.fit(train_X, train_y)
##
##y_pred = model_xgb.predict(test_X)
##y_pred = np.around(y_pred)
##
##meanDifference = np.mean(y_pred - max_floor_test_X)
##
##max_floor_ToPredict_Train = train_df.ix[(train_df['max_floor'].isnull() == True)][['floor', 'build_year']].values
##PredictedMissing_max_floor_Train = model_xgb.predict(max_floor_ToPredict_Train)
##PredictedMissing_max_floor_Train = np.around(PredictedMissing_max_floor_Train)
##
##max_floor_ToPredict_Test = test_df.ix[(test_df['max_floor'].isnull() == True)][['floor', 'build_year']].values
##PredictedMissing_max_floor_Test = model_xgb.predict(max_floor_ToPredict_Test)
##PredictedMissing_max_floor_Test = np.around(PredictedMissing_max_floor_Test)
##
##
##countOfPredictedValue_train = 0
##for value in train_data_max_floor_NaN_index:
##    train_df.iloc[value,5] = PredictedMissing_max_floor_Train[countOfPredictedValue_train]
##    countOfPredictedValue_train = countOfPredictedValue_train + 1
##
##countOfPredictedValue_test = 0
##for value in test_data_max_floor_NaN_index:
##    test_df.iloc[value,5] = PredictedMissing_max_floor_Test[countOfPredictedValue_test]
##    countOfPredictedValue_test = countOfPredictedValue_test + 1
##
##print("ADDED MISSING max_floor VALUES")
##
####***************************FIX full_sq: add median-+5**********************************************************************************************************
##
##print("ADDING MISSING full_sq VALUES...")
##
##train_df['full_sq'].ix[(train_df['full_sq'].isnull() == True)] = train_df['full_sq'].ix[(train_df['full_sq'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['full_sq'].median()-5, all_data_without_NaN['full_sq'].median()+5))
##
##test_df['full_sq'].ix[(test_df['full_sq'].isnull() == True)] = test_df['full_sq'].ix[(test_df['full_sq'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['full_sq'].median()-5, all_data_without_NaN['full_sq'].median()+5))
##
##print("ADDED MISSING full_sq VALUES")
##
####***************************FIX kitch_sq with: full_sq, max_floor(0.73), floor(0.38)**********************************************************************************************************
##
##print("ADDING MISSING kitch_sq VALUES...")
##
##train_df_without_NaN = train_df.dropna()
##test_df_without_NaN = test_df.dropna()
##all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
##                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))
##
##DataMostCorrelatedTo_kitch_sq = all_data_without_NaN[['floor', 'max_floor', 'full_sq']]
##
##train_X = DataMostCorrelatedTo_kitch_sq.values
####print(train_X.shape)##(7859, 3)
##train_y = all_data_without_NaN['kitch_sq']
####print(train_y.values.shape)##((7859, 3)
##test_X = all_data.ix[(all_data['kitch_sq'].isnull() == False)][['floor', 'max_floor', 'full_sq']].values
####print(test_X.shape)##(20088, 3)
##kitch_sq_test_X = all_data.ix[(all_data['kitch_sq'].isnull() == False)][['kitch_sq']].values
####print(kitch_sq_test_X.shape)##(20088, 1)
##
##dtrain = xgb.DMatrix(train_X, label = train_y)
##
##earlyStopRounds = 10
##currentParams = {"max_depth":4, "gamma": 0.1,
##                 "eta":0.01, "colsample_bytree":0.75,
##                 "subsample": 0.75, 'objective': 'reg:linear',
##                 'min_child_weight':1}
##SEED = 0
##
####model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
####        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
####print("ERROR: ", model.values[model.shape[0]-1][0])
####print("BEST ROUND: ", model.shape[0])
##
####ERROR:  1.8110122
####BEST ROUND:  410
##
##best_num_boost_round = 475
##best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))
##
##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=best_num_boost_round,
##    max_depth=4,
##    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
##    gamma=0.1)
##
##model_xgb.fit(train_X, train_y)
##
##y_pred = model_xgb.predict(test_X)
##y_pred = np.around(y_pred)
##
##meanDifference = np.mean(y_pred - max_floor_test_X)
##
##kitch_sq_ToPredict_Train = train_df.ix[(train_df['kitch_sq'].isnull() == True)][['floor', 'max_floor', 'full_sq']].values
##PredictedMissing_kitch_sq_Train = model_xgb.predict(kitch_sq_ToPredict_Train)
##PredictedMissing_kitch_sq_Train = np.around(PredictedMissing_kitch_sq_Train - meanDifference)
##
##kitch_sq_ToPredict_Test = test_df.ix[(test_df['kitch_sq'].isnull() == True)][['floor', 'max_floor', 'full_sq']].values
##PredictedMissing_kitch_sq_Test = model_xgb.predict(kitch_sq_ToPredict_Test)
##PredictedMissing_kitch_sq_Test = np.around(PredictedMissing_kitch_sq_Test - meanDifference)
##
##mean_kitch_sq_before_train = train_df['kitch_sq'].mean()
##mean_kitch_sq_before_test = test_df['kitch_sq'].mean()
##
##
##
##countOfPredictedValue_train = 0
##for value in train_data_kitch_sq_NaN_index:
####    print(value)
##    train_df.iloc[value,9] = PredictedMissing_kitch_sq_Train[countOfPredictedValue_train] 
##    countOfPredictedValue_train = countOfPredictedValue_train + 1
##
##countOfPredictedValue_test = 0
##for value in test_data_kitch_sq_NaN_index:
####    print(value)
##    test_df.iloc[value,9] = PredictedMissing_kitch_sq_Test[countOfPredictedValue_test]
##    countOfPredictedValue_test = countOfPredictedValue_test + 1
##
##
##
##mean_kitch_sq_after_train = train_df['kitch_sq'].mean()
##mean_kitch_sq_after_test = test_df['kitch_sq'].mean()
##
##DifferenceOfMean_train = mean_kitch_sq_before_train - mean_kitch_sq_after_train
##DifferenceOfMean_test = mean_kitch_sq_before_test - mean_kitch_sq_after_test
##
##
##
##threshold = -1
##
##while DifferenceOfMean_train < threshold:
##    countOfPredictedValue_train = 0
##    for value in train_data_kitch_sq_NaN_index:
##    ##    print(value)
##        train_df.iloc[value,9] = train_df.iloc[value,9] + DifferenceOfMean_train
##        countOfPredictedValue_train = countOfPredictedValue_train + 1
##    mean_kitch_sq_after_train = train_df['kitch_sq'].mean()
##    DifferenceOfMean_train = mean_kitch_sq_before_train - mean_kitch_sq_after_train
##
##while DifferenceOfMean_test < threshold:
##
##    countOfPredictedValue_test = 0
##    for value in test_data_kitch_sq_NaN_index:
##    ##    print(value)
##        test_df.iloc[value,9] = test_df.iloc[value,9] + DifferenceOfMean_test
##        countOfPredictedValue_test = countOfPredictedValue_test + 1
##    mean_kitch_sq_after_test = test_df['kitch_sq'].mean()
##    DifferenceOfMean_test = mean_kitch_sq_before_test - mean_kitch_sq_after_test
##    
####    
##
##
##
##print("ADDED MISSING kitch_sq VALUES")
##
####***************************FIX num_room with: full_sq, kitch_sq**********************************************************************************************************
##
##print("ADDING MISSING num_room VALUES...")
##
##train_df_without_NaN = train_df.dropna()
##test_df_without_NaN = test_df.dropna()
##all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
##                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))
##
##DataMostCorrelatedTo_num_room = all_data_without_NaN[['full_sq', 'kitch_sq']]
##
##train_X = DataMostCorrelatedTo_num_room.values
####print(train_X.shape)##(7859, 3)
##train_y = all_data_without_NaN['num_room']
####print(train_y.values.shape)##((7859, 3)
##test_X = all_data.ix[(all_data['num_room'].isnull() == False)][['full_sq', 'kitch_sq']].values
####print(test_X.shape)##(20088, 3)
##num_room_test_X = all_data.ix[(all_data['num_room'].isnull() == False)][['num_room']].values
####print(kitch_sq_test_X.shape)##(20088, 1)
##
##dtrain = xgb.DMatrix(train_X, label = train_y)
##
##earlyStopRounds = 10
##currentParams = {"max_depth":4, "gamma": 0.1,
##                 "eta":0.01, "colsample_bytree":0.75,
##                 "subsample": 0.75, 'objective': 'reg:linear',
##                 'min_child_weight':1}
##SEED = 0
##
####model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
####        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
####print("ERROR: ", model.values[model.shape[0]-1][0])
####print("BEST ROUND: ", model.shape[0])
##
####ERROR:  0.4173778
####BEST ROUND:  1216
##
##best_num_boost_round = 1216
##best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))
##
##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=best_num_boost_round,
##    max_depth=4,
##    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
##    gamma=0.1)
##
##model_xgb.fit(train_X, train_y)
##
##y_pred = model_xgb.predict(test_X)
##y_pred = np.around(y_pred)
##
##meanDifference = np.mean(y_pred - num_room_test_X)
##
##num_room_ToPredict_Train = train_df.ix[(train_df['num_room'].isnull() == True)][['full_sq', 'kitch_sq']].values
##PredictedMissing_num_room_Train = model_xgb.predict(num_room_ToPredict_Train)
##PredictedMissing_num_room_Train = np.around(PredictedMissing_num_room_Train)
##
##num_room_ToPredict_Test = test_df.ix[(test_df['num_room'].isnull() == True)][['full_sq', 'kitch_sq']].values
##PredictedMissing_num_room_Test = model_xgb.predict(num_room_ToPredict_Test)
##PredictedMissing_num_room_Test = np.around(PredictedMissing_num_room_Test)
##
##mean_num_room_before_train = train_df['num_room'].mean()
##mean_num_room_before_test = test_df['num_room'].mean()
##
##
##countOfPredictedValue_train = 0
##for value in train_data_num_room_NaN_index:
####    print(value)
##    train_df.iloc[value,8] = PredictedMissing_num_room_Train[countOfPredictedValue_train] 
##    countOfPredictedValue_train = countOfPredictedValue_train + 1
##
##countOfPredictedValue_test = 0
##for value in test_data_num_room_NaN_index:
####    print(value)
##    test_df.iloc[value,8] = PredictedMissing_num_room_Test[countOfPredictedValue_test]
##    countOfPredictedValue_test = countOfPredictedValue_test + 1
##
##
##
##mean_num_room_after_train = train_df['num_room'].mean()
##mean_num_room_after_test = test_df['num_room'].mean()
##
##DifferenceOfMean_train = mean_num_room_before_train - mean_num_room_after_train
##DifferenceOfMean_test = mean_num_room_before_test - mean_num_room_after_test
##
####print(DifferenceOfMean_train)##0.117363361075
####print(DifferenceOfMean_test)##0.000228077811298
##
##
##print("ADDED MISSING num_room VALUES")
##
####***************************FIX life_sq with: full_sq, num_room**********************************************************************************************************
##
##print("ADDING MISSING life_sq VALUES...")
##
##train_df_without_NaN = train_df.dropna()
##test_df_without_NaN = test_df.dropna()
##all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
##                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))
##
##
##DataMostCorrelatedTo_life_sq = all_data_without_NaN[['full_sq', 'num_room']]
##
##train_X = DataMostCorrelatedTo_life_sq.values
####print(train_X.shape)##(7859, 3)
##train_y = all_data_without_NaN['life_sq']
####print(train_y.values.shape)##((7859, 3)
##test_X = all_data.ix[(all_data['life_sq'].isnull() == False)][['full_sq', 'num_room']].values
####print(test_X.shape)##(20088, 3)
##life_sq_test_X = all_data.ix[(all_data['life_sq'].isnull() == False)][['life_sq']].values
####print(kitch_sq_test_X.shape)##(20088, 1)
##
##dtrain = xgb.DMatrix(train_X, label = train_y)
##
##earlyStopRounds = 10
##currentParams = {"max_depth":4, "gamma": 0.1,
##                 "eta":0.01, "colsample_bytree":0.75,
##                 "subsample": 0.75, 'objective': 'reg:linear',
##                 'min_child_weight':1}
##SEED = 0
##
####model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
####        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
####print("ERROR: ", model.values[model.shape[0]-1][0])
####print("BEST ROUND: ", model.shape[0])
##
####ERROR:  7.9550282
####BEST ROUND:  617
##
##best_num_boost_round = 617
##best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))
##
##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=best_num_boost_round,
##    max_depth=4,
##    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
##    gamma=0.1)
##
##model_xgb.fit(train_X, train_y)
##
##y_pred = model_xgb.predict(test_X)
##y_pred = np.around(y_pred)
##
##meanDifference = np.mean(y_pred - life_sq_test_X)
##
##life_sq_ToPredict_Train = train_df.ix[(train_df['life_sq'].isnull() == True)][['full_sq', 'num_room']].values
##PredictedMissing_life_sq_Train = model_xgb.predict(life_sq_ToPredict_Train)
##PredictedMissing_life_sq_Train = np.around(PredictedMissing_life_sq_Train)
##
##life_sq_ToPredict_Test = test_df.ix[(test_df['life_sq'].isnull() == True)][['full_sq', 'num_room']].values
##PredictedMissing_life_sq_Test = model_xgb.predict(life_sq_ToPredict_Test)
##PredictedMissing_life_sq_Test = np.around(PredictedMissing_life_sq_Test)
##
##
##
##
##
##mean_life_sq_before_train = train_df['life_sq'].mean()
##mean_life_sq_before_test = test_df['life_sq'].mean()
##
##
##
##countOfPredictedValue_train = 0
##for value in train_data_life_sq_NaN_index:
####    print(value)
##    train_df.iloc[value,3] = PredictedMissing_life_sq_Train[countOfPredictedValue_train] 
##    countOfPredictedValue_train = countOfPredictedValue_train + 1
##
##countOfPredictedValue_test = 0
##for value in test_data_life_sq_NaN_index:
####    print(value)
##    test_df.iloc[value,3] = PredictedMissing_life_sq_Test[countOfPredictedValue_test]
##    countOfPredictedValue_test = countOfPredictedValue_test + 1
##
##
##
##mean_life_sq_after_train = train_df['life_sq'].mean()
##mean_life_sq_after_test = test_df['life_sq'].mean()
##
##DifferenceOfMean_train = mean_life_sq_before_train - mean_life_sq_after_train
##DifferenceOfMean_test = mean_life_sq_before_test - mean_life_sq_after_test
##
####
####print("ADDED MISSING life_sq VALUES")
##
##
##
##
##
####***************************Give everything a random for now(20/06/17, 6:15)**********************************************************************************************************
##
##print("ADDING SOME MISSING VALUES WITH RANDOM...")
##
####train_df['life_sq'].ix[(train_df['life_sq'].isnull() == True)] = train_df['life_sq'].ix[(train_df['life_sq'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['life_sq'].median()-3), int(all_data_without_NaN['life_sq'].median()+3)))
####test_df['life_sq'].ix[(test_df['life_sq'].isnull() == True)] = test_df['life_sq'].ix[(test_df['life_sq'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['life_sq'].median()-3), int(all_data_without_NaN['life_sq'].median()+3)))
####  
####train_df['kitch_sq'].ix[(train_df['kitch_sq'].isnull() == True)] = train_df['kitch_sq'].ix[(train_df['kitch_sq'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['kitch_sq'].median()-3), int(all_data_without_NaN['kitch_sq'].median()+3)))
####test_df['kitch_sq'].ix[(test_df['kitch_sq'].isnull() == True)] = test_df['kitch_sq'].ix[(test_df['kitch_sq'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['kitch_sq'].median()-3), int(all_data_without_NaN['kitch_sq'].median()+3)))
####
####train_df['num_room'].ix[(train_df['num_room'].isnull() == True)] = train_df['num_room'].ix[(train_df['num_room'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['num_room'].median()-1, all_data_without_NaN['num_room'].median()+1))
####test_df['num_room'].ix[(test_df['num_room'].isnull() == True)] = test_df['num_room'].ix[(test_df['num_room'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['num_room'].median()-1, all_data_without_NaN['num_room'].median()+1))
##
####count    12084.000000
####mean      1384.542370
####std       1158.742568
####min         30.000000
####25%        350.000000
####50%       1084.000000
####75%       1970.000000
####max       4849.000000
####Name: hospital_beds_raion, dtype: float64
##
##train_df['hospital_beds_raion'].ix[(train_df['hospital_beds_raion'].isnull() == True)] = train_df['hospital_beds_raion'].ix[(train_df['hospital_beds_raion'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['hospital_beds_raion'].median()-400, all_data_without_NaN['hospital_beds_raion'].median()+400))
##test_df['hospital_beds_raion'].ix[(test_df['hospital_beds_raion'].isnull() == True)] = test_df['hospital_beds_raion'].ix[(test_df['hospital_beds_raion'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['hospital_beds_raion'].median()-400, all_data_without_NaN['hospital_beds_raion'].median()+400))
##
##train_df['state'].ix[(train_df['state'].isnull() == True)] = train_df['state'].ix[(train_df['state'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['state'].median()-1, all_data_without_NaN['state'].median()+1))
##test_df['state'].ix[(test_df['state'].isnull() == True)] = test_df['state'].ix[(test_df['state'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['state'].median()-1, all_data_without_NaN['state'].median()+1))
##
##cafe_cols = [col for col in all_data.columns if 'cafe' in col]
##for value in cafe_cols:
####    train_df[value].ix[(train_df[value].isnull() == True)] = train_df[value].ix[(train_df[value].isnull() == True)].apply(lambda v: randint(0, int(all_data_without_NaN[value].median())))
####    test_df[value].ix[(test_df[value].isnull() == True)] = test_df[value].ix[(test_df[value].isnull() == True)].apply(lambda v: randint(0, int(all_data_without_NaN[value].median())))
##    train_df[value].ix[(train_df[value].isnull() == True)] = train_df[value].ix[(train_df[value].isnull() == True)].fillna(0)
##    test_df[value].ix[(test_df[value].isnull() == True)] = test_df[value].ix[(test_df[value].isnull() == True)].fillna(0)
##    
##build_count_cols = [col for col in all_data.columns if 'build_count' in col]
##for value in build_count_cols:
##    train_df[value].ix[(train_df[value].isnull() == True)] = train_df[value].ix[(train_df[value].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN[value].median()-5), int(all_data_without_NaN[value].median()+5)))
##    test_df[value].ix[(test_df[value].isnull() == True)] = test_df[value].ix[(test_df[value].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN[value].median()-5), int(all_data_without_NaN[value].median()+5)))
##
##
####count    8460.000000
####mean     1227.586061
####std       492.383027
####min       500.000000
####25%      1000.000000
####50%      1166.670000
####75%      1500.000000
####max      6000.000000
####Name: cafe_sum_500_max_price_avg, dtype: float64
##
####train_df['cafe_sum_500_max_price_avg'].ix[(train_df['cafe_sum_500_max_price_avg'].isnull() == True)] = train_df['cafe_sum_500_max_price_avg'].ix[(train_df['cafe_sum_500_max_price_avg'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['cafe_sum_500_max_price_avg'].median()-550), int(all_data_without_NaN['cafe_sum_500_max_price_avg'].median()+550)))
####test_df['cafe_sum_500_max_price_avg'].ix[(test_df['cafe_sum_500_max_price_avg'].isnull() == True)] = test_df['cafe_sum_500_max_price_avg'].ix[(test_df['cafe_sum_500_max_price_avg'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['cafe_sum_500_max_price_avg'].median()-550), int(all_data_without_NaN['cafe_sum_500_max_price_avg'].median()+550)))
##
####count    8460.000000
####mean      729.214528
####std       325.047879
####min       300.000000
####25%       500.000000
####50%       672.730000
####75%       942.860000
####max      4000.000000
####Name: cafe_sum_500_min_price_avg, dtype: float64
##
####train_df['cafe_sum_500_min_price_avg'].ix[(train_df['cafe_sum_500_min_price_avg'].isnull() == True)] = train_df['cafe_sum_500_min_price_avg'].ix[(train_df['cafe_sum_500_min_price_avg'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['cafe_sum_500_min_price_avg'].median()-300), int(all_data_without_NaN['cafe_sum_500_min_price_avg'].median()+300)))
####test_df['cafe_sum_500_min_price_avg'].ix[(test_df['cafe_sum_500_min_price_avg'].isnull() == True)] = test_df['cafe_sum_500_min_price_avg'].ix[(test_df['cafe_sum_500_min_price_avg'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['cafe_sum_500_min_price_avg'].median()-300), int(all_data_without_NaN['cafe_sum_500_min_price_avg'].median()+300)))
####
##
##print("ADDED SOME MISSING VALUES WITH RANDOM")
##
##
######***************************Predict school_quota with: raion_popul,work_female,work_all,0_17_female,children_preschool,0_6_all,0_17_all,0_6_male,0_13_all**********************************************************************************************************
##print("ADDING MISSING school_quota...")
##
##DataMostCorrelatedTo_school_quota = all_data_without_NaN[['raion_popul','work_female','work_all','0_17_female','children_preschool','0_6_all','0_17_all','0_6_male','0_13_all']]
##train_X = DataMostCorrelatedTo_school_quota.values
##train_y = all_data_without_NaN['school_quota']
##test_X = all_data.ix[(all_data['school_quota'].isnull() == False)][['raion_popul','work_female','work_all','0_17_female','children_preschool','0_6_all','0_17_all','0_6_male','0_13_all']].values
##school_quota_test_X = all_data.ix[(all_data['school_quota'].isnull() == False)][['school_quota']].values
##
##dtrain = xgb.DMatrix(train_X, label = train_y)
##
##earlyStopRounds = 10
##currentParams = {"max_depth":4, "gamma": 0.1,
##                 "eta":0.01, "colsample_bytree":0.75,
##                 "subsample": 0.75, 'objective': 'reg:linear',
##                 'min_child_weight':1}
##SEED = 0
##
####model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
####        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
####print("ERROR: ", model.values[model.shape[0]-1][0])
####print("BEST ROUND: ", model.shape[0])
##
##best_num_boost_round = 6031##6031
##best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))
##
##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=best_num_boost_round,
##    max_depth=4,
##    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
##    gamma=0.1)
##
##model_xgb.fit(train_X, train_y)
##y_pred = model_xgb.predict(test_X)
##y_pred = np.around(y_pred)
##
##meanDifference = np.mean(y_pred - school_quota_test_X)
##
##school_quota_ToPredict_Train = train_df.ix[(train_df['school_quota'].isnull() == True)][['raion_popul','work_female','work_all','0_17_female','children_preschool','0_6_all','0_17_all','0_6_male','0_13_all']].values
##PredictedMissing_school_quota_Train = model_xgb.predict(school_quota_ToPredict_Train)
##PredictedMissing_school_quota_Train = np.around(PredictedMissing_school_quota_Train - meanDifference)
##
##school_quota_ToPredict_Test = test_df.ix[(test_df['school_quota'].isnull() == True)][['raion_popul','work_female','work_all','0_17_female','children_preschool','0_6_all','0_17_all','0_6_male','0_13_all']].values
##PredictedMissing_school_quota_Test = model_xgb.predict(school_quota_ToPredict_Test)
##PredictedMissing_school_quota_Test = np.around(PredictedMissing_school_quota_Test - meanDifference)
##
##
##
##
####
##mean_school_quota_before_train = train_df['school_quota'].mean()
##mean_school_quota_before_test = test_df['school_quota'].mean()
##
####print(mean_school_quota_before_train)
####print(mean_school_quota_before_test)
##
##countOfPredictedValue_train = 0
##for value in train_data_school_quota_NaN_index:
####    print(value)
##    train_df.iloc[value,21] = PredictedMissing_school_quota_Train[countOfPredictedValue_train] 
##    countOfPredictedValue_train = countOfPredictedValue_train + 1
##
##countOfPredictedValue_test = 0
##for value in test_data_school_quota_NaN_index:
####    print(value)
##    test_df.iloc[value,21] = PredictedMissing_school_quota_Test[countOfPredictedValue_test]
##    countOfPredictedValue_test = countOfPredictedValue_test + 1
##
##
##mean_school_quota_after_train = train_df['school_quota'].mean()
##mean_school_quota_after_test = test_df['school_quota'].mean()
##
####print(mean_school_quota_after_train)
####print(mean_school_quota_after_test)
##
##DifferenceOfMean_train = mean_school_quota_before_train - mean_school_quota_after_train
##DifferenceOfMean_test = mean_school_quota_before_test - mean_school_quota_after_test
##
##
##
##while DifferenceOfMean_train > 100:
##    countOfPredictedValue_train = 0
##    for value in train_data_school_quota_NaN_index:
##    ##    print(value)
##        train_df.iloc[value,21] = train_df.iloc[value,21] + DifferenceOfMean_train
##        countOfPredictedValue_train = countOfPredictedValue_train + 1
##    mean_school_quota_after_train = train_df['school_quota'].mean()
##    DifferenceOfMean_train = mean_school_quota_before_train - mean_school_quota_after_train
##
##while DifferenceOfMean_test > 100:
##
##    countOfPredictedValue_test = 0
##    for value in test_data_school_quota_NaN_index:
##    ##    print(value)
##        test_df.iloc[value,21] = test_df.iloc[value,21] + DifferenceOfMean_test
##        countOfPredictedValue_test = countOfPredictedValue_test + 1
##    mean_school_quota_after_test = test_df['school_quota'].mean()
##    DifferenceOfMean_test = mean_school_quota_before_test - mean_school_quota_after_test
##    
##
##print("ADDED MISSING school_quota")
##
##
######***************************Predict preschool_quota with: work_female, raion_popul, work_all, work_male, children_preschool, 0_6_all, 0_6_male**********************************************************************************************************
##
##print("ADDING MISSING preschool_quota...")
##
##DataMostCorrelatedTo_preschool_quota = all_data_without_NaN[['work_female', 'raion_popul', 'work_all', 'work_male', 'children_preschool', '0_6_all', '0_6_male']]
##
##train_X = DataMostCorrelatedTo_preschool_quota.values
####print(train_X.shape)##(7859, 8)
##train_y = all_data_without_NaN['preschool_quota']
####print(train_y.values.shape)##(7859,)
##test_X = all_data.ix[(all_data['preschool_quota'].isnull() == False)][['work_female', 'raion_popul', 'work_all', 'work_male', 'children_preschool', '0_6_all', '0_6_male']].values
####print(test_X.shape)##(29849, 8)
##preschool_quota_test_X = all_data.ix[(all_data['preschool_quota'].isnull() == False)][['preschool_quota']].values
####print(preschool_quota_test_X.shape)##(29849, 1)
##
##dtrain = xgb.DMatrix(train_X, label = train_y)
##
##earlyStopRounds = 10
##currentParams = {"max_depth":4, "gamma": 0.1,
##                 "eta":0.01, "colsample_bytree":0.75,
##                 "subsample": 0.75, 'objective': 'reg:linear',
##                 'min_child_weight':1}
##SEED = 0
##
####model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
####        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
####print("ERROR: ", model.values[model.shape[0]-1][0])
####print("BEST ROUND: ", model.shape[0])
##
####ERROR:  0.0307786
####BEST ROUND:  7047
##
##best_num_boost_round = 7047
##best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))
##
##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=best_num_boost_round,
##    max_depth=4,
##    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
##    gamma=0.1)
##
##model_xgb.fit(train_X, train_y)
##y_pred = model_xgb.predict(test_X)
##y_pred = np.around(y_pred)
##
##meanDifference = np.mean(y_pred - preschool_quota_test_X)
##
##
##preschool_quota_ToPredict_Train = train_df.ix[(train_df['preschool_quota'].isnull() == True)][['work_female', 'raion_popul', 'work_all', 'work_male', 'children_preschool', '0_6_all', '0_6_male']].values
##PredictedMissing_preschool_quota_Train = model_xgb.predict(preschool_quota_ToPredict_Train)
##PredictedMissing_preschool_quota_Train = np.around(PredictedMissing_preschool_quota_Train-meanDifference)
##
##preschool_quota_ToPredict_Test = test_df.ix[(test_df['preschool_quota'].isnull() == True)][['work_female', 'raion_popul', 'work_all', 'work_male', 'children_preschool', '0_6_all', '0_6_male']].values
##PredictedMissing_preschool_quota_Test = model_xgb.predict(preschool_quota_ToPredict_Test)
##PredictedMissing_preschool_quota_Test = np.around(PredictedMissing_preschool_quota_Test-meanDifference)
##
####print(train_df['preschool_quota'].unique())
####print(test_df['preschool_quota'].unique())
##
##
##mean_preschool_quota_before_train = train_df['preschool_quota'].mean()
##mean_preschool_quota_before_test = test_df['preschool_quota'].mean()
##
##countOfPredictedValue_train = 0
##for value in train_data_preschool_quota_NaN_index:
####    print(value)
##    train_df.iloc[value,18] = PredictedMissing_preschool_quota_Train[countOfPredictedValue_train] 
##    countOfPredictedValue_train = countOfPredictedValue_train + 1
##
##countOfPredictedValue_test = 0
##for value in test_data_preschool_quota_NaN_index:
####    print(value)
##    test_df.iloc[value,18] = PredictedMissing_preschool_quota_Test[countOfPredictedValue_test]
##    countOfPredictedValue_test = countOfPredictedValue_test + 1
##
####print(train_df['preschool_quota'].unique())
####print(test_df['preschool_quota'].unique())
##
##mean_preschool_quota_after_train = train_df['preschool_quota'].mean()
##mean_preschool_quota_after_test = test_df['preschool_quota'].mean()
##
##DifferenceOfMean_train = mean_preschool_quota_before_train - mean_preschool_quota_after_train
##DifferenceOfMean_test = mean_preschool_quota_before_test - mean_preschool_quota_after_test
##
##
##countOfPredictedValue_train = 0
##for value in train_data_preschool_quota_NaN_index:
####    print(value)
##    train_df.iloc[value,18] = train_df.iloc[value,18] + DifferenceOfMean_train
##    countOfPredictedValue_train = countOfPredictedValue_train + 1
##
##countOfPredictedValue_test = 0
##for value in test_data_preschool_quota_NaN_index:
####    print(value)
##    test_df.iloc[value,18] = test_df.iloc[value,18] + DifferenceOfMean_test
##    countOfPredictedValue_test = countOfPredictedValue_test + 1
##
##
##while DifferenceOfMean_train > 100:
##    countOfPredictedValue_train = 0
##    for value in train_data_preschool_quota_NaN_index:
##    ##    print(value)
##        train_df.iloc[value,18] = train_df.iloc[value,18] + DifferenceOfMean_train
##        countOfPredictedValue_train = countOfPredictedValue_train + 1
##    mean_preschool_quota_after_train = train_df['preschool_quota'].mean()
##    DifferenceOfMean_train = mean_preschool_quota_before_train - mean_preschool_quota_after_train
##
##while DifferenceOfMean_test > 100:
##
##    countOfPredictedValue_test = 0
##    for value in test_data_preschool_quota_NaN_index:
##    ##    print(value)
##        test_df.iloc[value,18] = test_df.iloc[value,18] + DifferenceOfMean_test
##        countOfPredictedValue_test = countOfPredictedValue_test + 1
##    mean_preschool_quota_after_test = test_df['preschool_quota'].mean()
##    DifferenceOfMean_test = mean_preschool_quota_before_test - mean_preschool_quota_after_test
##
##
##print("ADDED MISSING preschool_quota")
##
##
##
##
##print("FIXED DATA BY PREDICTING MISSING DATA WITH EXISTING ONES")

##**********************************************FIXED DATA BY PREDICTING MISSING DATA WITH EXISTING ONES*********************************************###

##print(macro_df.describe())
##test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')
##train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')

all_data = pd.concat((train_df.loc[:,'timestamp':'market_count_5000'], test_df.loc[:,'timestamp':'market_count_5000']))
id_test = test_df['id']

# truncate the extreme values in price_doc #
ulimit = np.percentile(train_df.price_doc.values, 99)
llimit = np.percentile(train_df.price_doc.values, 1)
train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit






listOfDroppedColumns =['sub_area', 'green_zone_part', 'indust_part',
                       'thermal_power_plant_raion', 'incineration_raion','oil_chemistry_raion',
                       'radiation_raion','railroad_terminal_raion','big_market_raion','nuclear_reactor_raion'
                       ,'detention_facility_raion','full_all', 'male_f', 'female_f', 'young_all','work_all',
                       'ekder_all','0_6_all', '7_14_all','0_17_all','0_17_male','0_17_female','16_29_all',
                       '0_13_all','0_13_male','0_13_female','metro_min_avto','metro_min_walk','railroad_station_walk_min',
                       'railroad_station_avto_min','ID_railroad_station_avto','public_transport_station_min_walk',
                       'water_1line','ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal',
                       'ice_rink_km','basketball_km','swim_pool_km','detention_facility_km','eurrub',
                       'build_count_frame','build_count_1971-1995','build_count_block','build_count_wood',
                       'build_count_mix','build_count_1921-1945','build_count_panel','build_count_foam',
                       'build_count_slag','raion_build_count_with_builddate_info','build_count_monolith',
                       'build_count_before_1920','build_count_1946-1970','build_count_brick','raion_build_count_with_material_info',
                       'build_count_after_1995','material','cafe_avg_price_500','cafe_sum_500_min_price_avg','cafe_sum_500_max_price_avg',
                       'cafe_avg_price_1000','cafe_sum_1000_max_price_avg','cafe_sum_1000_min_price_avg','cafe_avg_price_1500'
                       'cafe_sum_1500_min_price_avg','cafe_sum_1500_max_price_avg']

listOfDroppedColumns2 =['ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal','sub_area'
                        ,'ID_railroad_station_avto','build_count_frame','build_count_1971-1995','build_count_block','build_count_wood',
                       'build_count_mix','build_count_1921-1945','build_count_panel','build_count_foam',
                       'build_count_slag','raion_build_count_with_builddate_info','build_count_monolith',
                       'build_count_before_1920','build_count_1946-1970','build_count_brick','raion_build_count_with_material_info',
                       'build_count_after_1995',
                       'thermal_power_plant_raion', 'incineration_raion','oil_chemistry_raion',
                       'radiation_raion','railroad_terminal_raion','big_market_raion','nuclear_reactor_raion'
                       ,'detention_facility_raion','full_all', 'male_f', 'female_f', 'young_all','work_all',
                       'ekder_all','0_6_all', '7_14_all','0_17_all','0_17_male','0_17_female','16_29_all',
                       '0_13_all','0_13_male','0_13_female','metro_min_avto','metro_min_walk','railroad_station_walk_min',
                       'railroad_station_avto_min','ID_railroad_station_avto','public_transport_station_min_walk','eurrub','ekder_all'
                        ,'cafe_avg_price_500','cafe_sum_500_min_price_avg','cafe_sum_500_max_price_avg',
                       'cafe_avg_price_1000','cafe_sum_1000_max_price_avg','cafe_sum_1000_min_price_avg','cafe_avg_price_1500',
                       'cafe_sum_1500_min_price_avg','cafe_sum_1500_max_price_avg','water_1line','big_road1_1line','railroad_1line',
                        'build_year'
                        ]

listOfDroppedColumns3 =['ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal',
                        ]

listOfLowestCorrelationColumns = []

## CATEGORICAL COLUMNS
##product_type
##sub_area: DELETED
##culture_objects_top_25
##thermal_power_plant_raion: DELETED
##incineration_raion: DELETED
##oil_chemistry_raion: DELETED
##radiation_raion: DELETED
##railroad_terminal_raion: DELETED
##big_market_raion: DELETED
##nuclear_reactor_raion: DELETED
##detention_facility_raion: DELETED
##water_1line: DELETED
##big_road1_1line: DELETED
##railroad_1line: DELETED
##ecology: CHANGED TO NUMERICAL
##child_on_acc_pre_school: NOT INCLUDED IN THE MACRO COLUMNS
##modern_education_share: NOT INCLUDED IN THE MACRO COLUMNS
##old_education_build_share: NOT INCLUDED IN THE MACRO COLUMNS

train_df.ecology[train_df.ecology == 'excellent'] = 4
train_df.ecology[train_df.ecology == 'good'] = 3
train_df.ecology[train_df.ecology == 'satisfactory'] = 2
train_df.ecology[train_df.ecology == 'poor'] = 1
train_df.ecology[train_df.ecology == 'no data'] = 0
train_df.ecology = train_df.ecology.astype('int')
train_df.ecology[train_df.ecology == 0] = train_df.ecology.median()


test_df.ecology[test_df.ecology == 'excellent'] = 4
test_df.ecology[test_df.ecology == 'good'] = 3
test_df.ecology[test_df.ecology == 'satisfactory'] = 2
test_df.ecology[test_df.ecology == 'poor'] = 1
test_df.ecology[test_df.ecology == 'no data'] = 0
test_df.ecology = test_df.ecology.astype('int')
test_df.ecology[test_df.ecology == 0] = test_df.ecology.median()

all_data.ecology[all_data.ecology == 'excellent'] = 4
all_data.ecology[all_data.ecology == 'good'] = 3
all_data.ecology[all_data.ecology == 'satisfactory'] = 2
all_data.ecology[all_data.ecology == 'poor'] = 1
all_data.ecology[all_data.ecology == 'no data'] = 0
all_data.ecology = all_data.ecology.astype('int')
all_data.ecology[all_data.ecology == 0] = all_data.ecology.median()









train_df_without_NAN = train_df.dropna()

train_df_without_NAN_no_dummies = train_df_without_NAN
train_df_without_NAN  = pd.get_dummies(train_df_without_NAN)

train_df = train_df.fillna(train_df.median())
test_df = test_df.fillna(test_df.median())
test_df_no_dummies = test_df
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)


all_data_without_NaN = all_data.dropna()
all_data_without_NaN = pd.get_dummies(all_data_without_NaN)

##foo = foo.apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)
print("filling NAN with random values....")
##all_data['floor'].ix[(all_data['floor'].isnull() == True)] = all_data['floor'].ix[(all_data['floor'].isnull() == True)].apply(lambda v: randint(all_data['floor'].median()-1, all_data['floor'].median()+1))
##all_data['max_floor'].ix[(all_data['max_floor'].isnull() == True)] = all_data['max_floor'].ix[(all_data['max_floor'].isnull() == True)].apply(lambda v: randint(all_data['max_floor'].median()-3, all_data['max_floor'].median()+3))
##all_data['state'].ix[(all_data['state'].isnull() == True)] = all_data['state'].ix[(all_data['state'].isnull() == True)].apply(lambda v: randint(all_data['state'].median()-1, all_data['state'].median()+1))
##all_data['num_room'].ix[(all_data['num_room'].isnull() == True)] = all_data['num_room'].ix[(all_data['num_room'].isnull() == True)].apply(lambda v: randint(all_data['num_room'].median()-1, all_data['num_room'].median()+1))
##all_data['material'].ix[(all_data['material'].isnull() == True)] = all_data['material'].ix[(all_data['material'].isnull() == True)].apply(lambda v: randint(all_data['material'].median()-1, all_data['material'].median()+1))
##all_data['build_year'].ix[(all_data['build_year'].isnull() == True)] = all_data['build_year'].ix[(all_data['build_year'].isnull() == True)].apply(lambda v: randint(all_data['build_year'].median()-15, all_data['build_year'].median()+5))
##all_data['kitch_sq'].ix[(all_data['kitch_sq'].isnull() == True)] = all_data['kitch_sq'].ix[(all_data['kitch_sq'].isnull() == True)].apply(lambda v: randint(all_data['kitch_sq'].median()-3, all_data['kitch_sq'].median()+3))

cafe_cols = [col for col in all_data.columns if 'cafe' in col]
for value in cafe_cols:
##    train_df[value].ix[(train_df[value].isnull() == True)] = train_df[value].ix[(train_df[value].isnull() == True)].apply(lambda v: randint(0, int(all_data_without_NaN[value].median())))
##    test_df[value].ix[(test_df[value].isnull() == True)] = test_df[value].ix[(test_df[value].isnull() == True)].apply(lambda v: randint(0, int(all_data_without_NaN[value].median())))
    all_data[value].ix[(all_data[value].isnull() == True)] = all_data[value].ix[(all_data[value].isnull() == True)].fillna(0)

print("filled NAN with random values")

##all_data = all_data.apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)
##all_data['max_floor'] = all_data['max_floor'].apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)
##all_data['state'] = all_data['state'].apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)
##all_data['kitch_sq'] = all_data['kitch_sq'].apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)
##all_data['num_room'] = all_data['num_room'].apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)
##all_data['material'] = all_data['material'].apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)
##all_data['build_year'] = all_data['build_year'].apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)

all_data['product_type']= all_data['product_type'].fillna('OwnerOccupier')
all_data['sub_area']= all_data['sub_area'].fillna('Severnoe Butovo')

all_data = all_data.fillna(all_data.median())

all_data = pd.get_dummies(all_data)










### year and month #
##train_df["yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month
test_df["yearmonth"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.month
train_df_without_NAN["yearmonth"] = train_df_without_NAN["timestamp"].dt.year*100 + train_df_without_NAN["timestamp"].dt.month
all_data["yearmonth"] = all_data["timestamp"].dt.year*100 + all_data["timestamp"].dt.month

### year and week #
##train_df["yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear
##test_df["yearweek"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.weekofyear
all_data["yearweek"] = all_data["timestamp"].dt.year*100 + all_data["timestamp"].dt.month

# year #
train_df["year"] = train_df["timestamp"].dt.year
test_df["year"] = test_df["timestamp"].dt.year
all_data["year"] = all_data["timestamp"].dt.year
all_data_without_NaN["year"] = all_data_without_NaN["timestamp"].dt.year
train_df_without_NAN["year"] = train_df_without_NAN["timestamp"].dt.year
##
### month of year #
##train_df["month_of_year"] = train_df["timestamp"].dt.month
##test_df["month_of_year"] = test_df["timestamp"].dt.month
all_data["month_of_year"] = all_data["timestamp"].dt.month

##
### week of year #
##train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear
##test_df["week_of_year"] = test_df["timestamp"].dt.weekofyear
all_data["week_of_year"] = all_data["timestamp"].dt.weekofyear

##
### day of week #
##train_df["day_of_week"] = train_df["timestamp"].dt.weekday
##test_df["day_of_week"] = test_df["timestamp"].dt.weekday
all_data["day_of_week"] = all_data["timestamp"].dt.weekday

##
##
# ratio of living area to full area #
train_df["ratio_life_sq_full_sq"] = train_df["life_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)
test_df["ratio_life_sq_full_sq"] = test_df["life_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)
train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"]<0] = 0
train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"]>1] = 1
test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"]<0] = 0
test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"]>1] = 1


train_df_without_NAN["ratio_life_sq_full_sq"] = train_df_without_NAN["life_sq"] / np.maximum(train_df_without_NAN["full_sq"].astype("float"),1)
train_df_without_NAN["ratio_life_sq_full_sq"].ix[train_df_without_NAN["ratio_life_sq_full_sq"]<0] = 0
train_df_without_NAN["ratio_life_sq_full_sq"].ix[train_df_without_NAN["ratio_life_sq_full_sq"]>1] = 1

##all_data["ratio_life_sq_full_sq"] = all_data["life_sq"] / np.maximum(all_data["full_sq"].astype("float"),1)
##all_data["ratio_life_sq_full_sq"].ix[all_data["ratio_life_sq_full_sq"]<0] = 0
##all_data["ratio_life_sq_full_sq"].ix[all_data["ratio_life_sq_full_sq"]>1] = 1

##
### ratio of kitchen area to living area #
##train_df["ratio_kitch_sq_life_sq"] = train_df["kitch_sq"] / np.maximum(train_df["life_sq"].astype("float"),1)
##test_df["ratio_kitch_sq_life_sq"] = test_df["kitch_sq"] / np.maximum(test_df["life_sq"].astype("float"),1)
##all_data["ratio_kitch_sq_life_sq"] = all_data["kitch_sq"] / np.maximum(all_data["life_sq"].astype("float"),1)

##train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]<0] = 0
##train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]>1] = 1
##test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]<0] = 0
##test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]>1] = 1
##all_data["ratio_kitch_sq_life_sq"].ix[all_data["ratio_kitch_sq_life_sq"]<0] = 0
##all_data["ratio_kitch_sq_life_sq"].ix[all_data["ratio_kitch_sq_life_sq"]>1] = 1

##
### ratio of kitchen area to full area #
##train_df["ratio_kitch_sq_full_sq"] = train_df["kitch_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)
##test_df["ratio_kitch_sq_full_sq"] = test_df["kitch_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)
##all_data["ratio_kitch_sq_full_sq"] = all_data["kitch_sq"] / np.maximum(all_data["full_sq"].astype("float"),1)

##train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]<0] = 0
##train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]>1] = 1
##test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]<0] = 0
##test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]>1] = 1
##all_data["ratio_kitch_sq_full_sq"].ix[all_data["ratio_kitch_sq_full_sq"]<0] = 0
##all_data["ratio_kitch_sq_full_sq"].ix[all_data["ratio_kitch_sq_full_sq"]>1] = 1

##
### floor of the house to the total number of floors in the house #
##train_df["ratio_floor_max_floor"] = train_df["floor"] / train_df["max_floor"].astype("float")
##test_df["ratio_floor_max_floor"] = test_df["floor"] / test_df["max_floor"].astype("float")
##all_data["ratio_floor_max_floor"] = all_data["floor"] / all_data["max_floor"].astype("float")

##
### num of floor from top #
##train_df["floor_from_top"] = train_df["max_floor"] - train_df["floor"]
##test_df["floor_from_top"] = test_df["max_floor"] - test_df["floor"]
##all_data["floor_from_top"] = all_data["max_floor"] - all_data["floor"]

##
train_df["extra_sq"] = train_df["full_sq"] - train_df["life_sq"]
test_df["extra_sq"] = test_df["full_sq"] - test_df["life_sq"]
train_df_without_NAN["extra_sq"] = train_df_without_NAN["full_sq"] - train_df_without_NAN["life_sq"]
all_data["extra_sq"] = all_data["full_sq"] - all_data["life_sq"]


##all_data["life_and_kitchen"] = all_data["kitch_sq"] + all_data["life_sq"]
test_df["life_and_kitchen"] = test_df["kitch_sq"] + test_df["life_sq"]

all_data["full_sq_times_state"] = all_data["full_sq"] * all_data["state"]
##all_data["num_room_times_state"] = all_data["num_room"] * all_data["state"]

##all_data["entertainment_Count_5000"] = all_data["trc_count_5000"] + all_data["cafe_count_5000"]+ all_data["sport_count_5000"]
##all_data["life_and_kitchen_times_state"] = (all_data["kitch_sq"] + all_data["life_sq"]) * all_data["state"]

##all_data["full_minus_life_and_kitchen"] = all_data["full_sq"] - all_data["life_sq"] - all_data["kitch_sq"]
##test_df["full_minus_life_and_kitchen"] = test_df["full_sq"] - test_df["life_sq"] - test_df["kitch_sq"]
##train_df_without_NAN["full_minus_life_and_kitchen"] = train_df_without_NAN["full_sq"] - train_df_without_NAN["life_sq"] - train_df_without_NAN["kitch_sq"]

all_data["full_life"] = all_data["full_sq"] + all_data["life_sq"]
all_data["full_life_kitch"] = all_data["full_sq"] + all_data["life_sq"] + all_data["kitch_sq"]

##
train_df["age_of_building"] = train_df["build_year"] - train_df["year"]
test_df["age_of_building"] = test_df["build_year"] - test_df["year"]
##all_data["age_of_building"] = all_data["build_year"] - all_data["year"]
##all_data_without_NaN["age_of_building"] = all_data_without_NaN["build_year"] - all_data_without_NaN["year"]
train_df_without_NAN["age_of_building"] = train_df_without_NAN["build_year"] - train_df_without_NAN["year"]


###Price of the house could also be affected by the availability of other houses at the same time period.
###So creating a count variable on the number of houses at the given time period might help.
##def add_count(df, group_col):
##    grouped_df = df.groupby(group_col)["id"].aggregate("count").reset_index()
##    grouped_df.columns = [group_col, "count_"+group_col]
##    df = pd.merge(df, grouped_df, on=group_col, how="left")
##    return df
##test_df = add_count(test_df, "yearmonth")
##train_df_without_NAN = add_count(train_df_without_NAN, "yearmonth")
##all_data = add_count(all_data, "yearmonth")

##
##train_df = add_count(train_df, "yearmonth")
##test_df = add_count(test_df, "yearmonth")
##
##train_df = add_count(train_df, "yearweek")
##test_df = add_count(test_df, "yearweek")
##
###Since schools generally play an important role in house hunting, let us create some variables around school.
##
##train_df["ratio_preschool"] = train_df["children_preschool"] / train_df["preschool_quota"].astype("float")
test_df["ratio_preschool"] = test_df["children_preschool"] / test_df["preschool_quota"].astype("float")
train_df_without_NAN["ratio_preschool"] = train_df_without_NAN["children_preschool"] / train_df_without_NAN["preschool_quota"].astype("float")
##all_data["ratio_preschool"] = all_data["children_preschool"] / all_data["preschool_quota"].astype("float")

##
##train_df["ratio_school"] = train_df["children_school"] / train_df["school_quota"].astype("float")
test_df["ratio_school"] = test_df["children_school"] / test_df["school_quota"].astype("float")
train_df_without_NAN["ratio_school"] = train_df_without_NAN["children_school"] / train_df_without_NAN["school_quota"].astype("float")
##all_data["ratio_school"] = all_data["children_school"] / all_data["school_quota"].astype("float")

##all_data["preschool_seat_available_rate"] = all_data["children_preschool"] - all_data["preschool_quota"]
##
##all_data["top20Highschool_to_numberOfHighSchool"] = all_data["school_education_centers_top_20_raion"] / all_data["school_education_centers_raion"]
##
##all_data["area_per_room"] = all_data["life_sq"] / all_data["num_room"]
##
all_data["cafe_count_5000_allPriceRange_sum"] = all_data["cafe_count_5000_price_500"] + all_data["cafe_count_5000_price_1000"] + all_data["cafe_count_5000_price_1500"] + all_data["cafe_count_5000_price_2500"] + all_data["cafe_count_5000_price_4000"] + all_data["cafe_count_5000_price_high"]   
##
##all_data["full_minus_lifeAndKitch"] = all_data["full_sq"] - all_data["life_sq"] - all_data["kitch_sq"]
##
##all_data["full_minus_lifeAndKitch_to_full"] = (all_data["full_sq"] - all_data["life_sq"] - all_data["kitch_sq"])/all_data["full_sq"]
##
##all_data["cafe_notNA_price_500"] = all_data["cafe_count_500"] - all_data["cafe_count_500_na_price"]
##
##all_data["cafe_notNA_price_1000"] = all_data["cafe_count_1000"] - all_data["cafe_count_1000_na_price"]
##
##all_data["cafe_notNA_price_1500"] = all_data["cafe_count_1500"] - all_data["cafe_count_1500_na_price"]
##
all_data["cafe_notNA_price_2000"] = all_data["cafe_count_2000"] - all_data["cafe_count_2000_na_price"]
##
all_data["cafe_notNA_price_3000"] = all_data["cafe_count_3000"] - all_data["cafe_count_3000_na_price"]
##
##all_data["cafe_notNA_price_5000"] = all_data["cafe_count_5000"] - all_data["cafe_count_5000_na_price"]

all_data = all_data.drop("timestamp", axis = 1)
all_data = all_data.drop(listOfDroppedColumns3)
all_data = all_data.drop(listOfLowestCorrelationColumns, axis = 1)

EngineeredFeatures = ['life_and_kitchen', 'full_sq_times_state', 'life_and_kitchen_times_state', 'extra_sq',
                     'num_room_times_state', "full_minus_life_and_kitchen",  
                      'age_of_building', "preschool_seat_available_rate",
                      ]

OriginalFeatures_Non_Category = ['full_sq', 'life_sq', 'kitch_sq', 'num_room', 'university_top_20_raion',
                                 'cafe_count_5000_price_1000', 'cafe_count_5000_price_1500', 'cafe_count_5000_price_2500',
                                 'sport_count_5000']

sub_area_cols = [col for col in all_data.columns if 'sub_area' in col]##<------it drops the rmse, but not much
product_type_cols = [col for col in all_data.columns if 'product_type' in col]##<------it drops the rmse by 0.002, better than sub_area_cols
radiation_raion_cols = [col for col in all_data.columns if 'radiation_raion' in col]##<------not helping, seems to get worse with it


allFeatures = EngineeredFeatures + OriginalFeatures_Non_Category

importantFeatures = ['full_sq', 'yearmonth', 'cafe_count_5000_price_high',
                     'cafe_count_5000_price_2500', 'full_sq_times_state', 'cafe_count_5000', 'cafe_count_2000',
                     'full_life_kitch', 'product_type_Investment', 'metro_min_avto', 'floor', 'sport_count_3000',
                     'ttk_km', 'full_life', 'cafe_count_3000', 'indust_part', 'max_floor', 'cafe_count_1500', 'bulvar_ring_km',
                     'metro_km_avto', 'railroad_km', 'leisure_count_3000', 'year', 'sport_objects_raion', 'extra_sq',
                     'cafe_count_2000_price_2500', 'school_education_centers_raion', 'cafe_count_1500_price_high',
                     'swim_pool_km', 'cafe_count_2000_price_1000', 'public_healthcare_km', 'cafe_count_5000_price_1000',
                     'university_km', 'age_of_building', 'detention_facility_km', 'cafe_count_5000_price_1500',
                     'ID_railroad_station_avto', 'shopping_centers_raion', 'nuclear_reactor_km', 'preschool_km',
                     'workplaces_km', 'cafe_count_3000_price_2500', 'cafe_count_3000_price_1500', 'trc_count_2000', 'state',
                     'exhibition_km', 'cafe_count_5000_na_price', 'cafe_count_1000_price_1000',
                     'trc_count_3000', 'prom_part_2000', 'prom_part_3000', 'park_km', 'office_sqm_5000',
                     'additional_education_km', 'build_year']

importantFeatures2 = ['full_sq', 'yearmonth', 'full_sq_times_state', 'cafe_count_5000_price_high',
                      'cafe_count_2000', 'detention_facility_km', 'cafe_count_5000_price_2500',
                      'full_life_kitch', 'cafe_count_1500', 'cafe_count_3000_price_1500',
                      'metro_km_avto', 'bulvar_ring_km', 'shopping_centers_raion', 'full_life',
                      'metro_min_avto', 'product_type_Investment', 'cafe_count_2000_price_2500',
                      'sport_count_3000', 'sport_objects_raion', 'cafe_count_5000_price_1500',
                      'cafe_count_3000', 'preschool_km', 'indust_part', 'yearweek',
                      'cafe_count_2000_price_1000', 'cafe_count_5000', 'public_healthcare_km',
                      'sport_count_2000', 'cafe_count_3000_price_2500', 'additional_education_km',
                      'theater_km', 'cafe_count_2000_price_500', 'cafe_count_2000_price_1500',
                      'bus_terminal_avto_km', 'green_part_3000', 'cafe_count_1000', 'exhibition_km',
                      'ratio_kitch_sq_full_sq', 'full_all', 'mosque_count_5000', 'floor']

importantFeatures3 = ['full_sq', 'yearmonth', 'cafe_count_5000_price_high',
                       'full_life_kitch', 'cafe_notNA_price_2000', 'detention_facility_km',
                       'cafe_count_3000_price_1500', 'num_room', 'full_sq_times_state', 'cafe_count_5000_price_2500',
                       'cafe_count_1500', 'yearweek', 'metro_min_avto', 'metro_km_avto', 'full_life', 'bulvar_ring_km',
                       'cafe_count_5000_allPriceRange_sum', 'shopping_centers_raion', 'floor', 'kitch_sq', 'sport_objects_raion',
                       'indust_part', 'cafe_count_2000_price_2500', 'additional_education_km', 'cafe_count_2000', 'product_type_Investment',
                       'preschool_km', 'sport_count_3000', 'swim_pool_km', 'cafe_count_3000', 'extra_sq', 'cafe_count_2000_price_1000',
                       'public_healthcare_km', 'cafe_count_2000_price_1500', 'cafe_notNA_price_3000', 'cafe_count_5000_price_1500', 'sport_count_2000',
                       'cafe_count_3000_price_2500', 'full_all', 'thermal_power_plant_km', 'theater_km',
                       'cafe_count_1000_price_1500', 'cafe_count_3000_price_1000', 'life_sq', 'max_floor', 'material', 
			'build_year', 'state', 'university_top_20_raion', 'university_km']

##all_data = all_data[allFeatures]
all_data = all_data[importantFeatures3]

train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
train_X = all_data[:train_df.shape[0]]

##train_X = train_df_without_NAN_no_dummies.drop(["id", "timestamp", "price_doc"], axis=1)
print(train_X.values.shape)

train_df['price_doc'] = np.log1p(train_df['price_doc'])
train_y = train_df.price_doc.values
##train_y = train_df_without_NAN_no_dummies.price_doc.values
print(train_y.shape)

##test_X = test_df_no_dummies.drop(["id", "timestamp"] , axis=1)
test_X = all_data[train_df.shape[0]:]
print(test_X.values.shape)

##****************************************************USE xgb.train instead of xgb.cv*****************************************************
##print("BEGIN USING xgb.train to get feature importance...")
##val_time = 201407
##dev_indices = np.where(train_X["yearmonth"]<val_time)
##val_indices = np.where(train_X["yearmonth"]>=val_time)
##dev_X = train_X.ix[dev_indices]
##val_X = train_X.ix[val_indices]
####train_y = np.log1p(train_df.price_doc.values)
##dev_y = train_y[dev_indices]
##val_y = train_y[val_indices]
##print(dev_X.shape, val_X.shape)
##
##
##xgtrain = xgb.DMatrix(dev_X, dev_y, feature_names=dev_X.columns)
##xgtest = xgb.DMatrix(val_X, val_y, feature_names=val_X.columns)
##
####xgtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns)
####xgtest = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns)
##
##watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
####watchlist = [ (xgtrain,'train')]
##
####
##xgb_params = {"max_depth":10, "gamma": 15.0,
##                 "eta":0.01, "colsample_bytree":0.9,
##                 "subsample": 0.9, 'objective': 'reg:linear',
##                 'min_child_weight':800}
####
##model = xgb.train(xgb_params, xgtrain, 100000000000, watchlist, early_stopping_rounds=50, verbose_eval=5)
##print(type(model.get_fscore().items()))
##
####items = sorted(model.get_fscore().items())
##
##
##sorted_by_importance = sorted(model.get_fscore().items(), key=operator.itemgetter(1), reverse=True)
####sorted_by_importance = sorted_by_importance.reverse()
##
##important_features = []
##
##for column_name, importance in sorted_by_importance:
##    if importance >= 5:
##        print(column_name)
##        important_features.append(column_name)
##        
##print(important_features)
####important_features = np.array(important_features)
####print(important_features)
##
####for value in sorted_by_importance:
####    print(value)
####
######feat_imp(train_X, model, 100)
####
####
##### plot the important features #
####fig, ax = plt.subplots(figsize=(12,18))##12,18
####xgb.plot_importance(model, height=0.8, ax=ax)
####plt.show()


    
##****************************************************USE xgb.cv*****************************************************





##
dtrain = xgb.DMatrix(train_X, label = train_y)

earlyStopRounds = 50
currentParams = {"max_depth":10, "gamma": 15.0,
                 "eta":0.01, "colsample_bytree":0.9,
                 "subsample": 0.9, 'objective': 'reg:linear',
                 'min_child_weight':800}

##currentParams = {"max_depth":10, "gamma": 10.0,
##                 "eta":0.01, "colsample_bytree":0.75,
##                 "subsample": 0.75, 'objective': 'reg:linear',
##                 'min_child_weight':800}
SEED = 0

print(currentParams)
model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
print("ERROR: ", model.values[model.shape[0]-1][0])
print("BEST ROUND: ", model.shape[0])

















































##***********************************************BELOW USES feature importance to train xgbClassifier***********************************
# split data into train and test sets
##print("BEGIN USING feature importance to train xgbClassifier")
##X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.20, random_state=7)
### fit model on all training data
##model = XGBClassifier()
##model.fit(X_train, y_train)
### make predictions for test data and evaluate
##y_pred = model.predict(X_test)
##predictions = [round(value) for value in y_pred]
##accuracy = accuracy_score(y_test, predictions)
##print("Accuracy: %.2f%%" % (accuracy * 100.0))
### Fit model using each importance as a threshold
##thresholds = sort(model.feature_importances_)
##for thresh in thresholds:
##	# select features using threshold
##	selection = SelectFromModel(model, threshold=thresh, prefit=True)
##	select_X_train = selection.transform(X_train)
##	# train model
##	selection_model = XGBClassifier()
##	selection_model.fit(select_X_train, y_train)
##	# eval model
##	select_X_test = selection.transform(X_test)
##	y_pred = selection_model.predict(select_X_test)
##	predictions = [round(value) for value in y_pred]
##	accuracy = accuracy_score(y_test, predictions)
##	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))




##***********************************************BELOW IS THE FINAL MODEL AFTER WE GET THE BEST SET OF PARAMETERS***********************************
##print("BEGIN GENERATING OUTPUT FILE......")
##earlyStopRounds = 50
##currentParams = {"max_depth":10, "gamma": 1.0,
##                 "eta":0.01, "colsample_bytree":0.9,
##                 "subsample": 0.9, 'objective': 'reg:linear',
##                 'min_child_weight':100}
##SEED = 0
####
####
##best_num_boost_round = model.shape[0]
##best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))
##
##
####best_num_boost_round = 838
####int((model.shape[0] - earlystop) / (1 - 1 / 5))
##
##
##
##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=best_num_boost_round,
##    max_depth=10,
##    learning_rate=0.01, subsample=0.9, colsample_bytree=0.9, min_child_weight=800, seed=SEED,
##    gamma=15.0)
##
##model_xgb.fit(train_X, train_y)
####model_xgb.fit(dev_X, dev_y)
##
##
#### plot the important features ##
######fig, ax = plt.subplots(figsize=(12,18))
######xgb.plot_importance(model_xgb, height=0.8, ax=ax)
######plt.show()
##
##ylog_pred = model_xgb.predict(test_X)
###np.expm1(model_xgb.predict(X_test))
####y_pred = np.exp(ylog_pred) - 1
##
##y_pred = np.expm1(ylog_pred)
##
##df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
##df_sub.to_csv('sub.csv', index=False)


print("END OF PROGRAM")



