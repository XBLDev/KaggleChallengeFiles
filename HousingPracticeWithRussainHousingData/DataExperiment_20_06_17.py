import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
##from sklearn import model_selection, preprocessing
import xgboost as xgb
import random
from random import randint
from random import uniform
from scipy import stats


color = sns.color_palette()
pd.options.mode.chained_assignment = None  # default='warn'

train_df = pd.read_csv("train_clean.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("test_clean.csv", parse_dates=['timestamp'])
all_data = pd.concat((train_df.loc[:,'timestamp':'market_count_5000'],
                      test_df.loc[:,'timestamp':'market_count_5000']))
train_df_without_NaN = train_df.dropna()
test_df_without_NaN = test_df.dropna()
all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))



all_data_life_sq_NaN = all_data['life_sq'].ix[(all_data['life_sq'].isnull() == True)]
all_data_life_sq_NaN_index = all_data['life_sq'].index[all_data['life_sq'].apply(np.isnan)]

train_data_life_sq_NaN = train_df['life_sq'].ix[(train_df['life_sq'].isnull() == True)]
train_data_life_sq_NaN_index = train_df['life_sq'].index[train_df['life_sq'].apply(np.isnan)]

test_data_life_sq_NaN = test_df['life_sq'].ix[(test_df['life_sq'].isnull() == True)]
test_data_life_sq_NaN_index = test_df['life_sq'].index[test_df['life_sq'].apply(np.isnan)]


all_data_num_room_NaN = all_data['num_room'].ix[(all_data['num_room'].isnull() == True)]
all_data_num_room_NaN_index = all_data['num_room'].index[all_data['num_room'].apply(np.isnan)]

train_data_num_room_NaN = train_df['num_room'].ix[(train_df['num_room'].isnull() == True)]
train_data_num_room_NaN_index = train_df['num_room'].index[train_df['num_room'].apply(np.isnan)]

test_data_num_room_NaN = test_df['num_room'].ix[(test_df['num_room'].isnull() == True)]
test_data_num_room_NaN_index = test_df['num_room'].index[test_df['num_room'].apply(np.isnan)]


all_data_kitch_sq_NaN = all_data['kitch_sq'].ix[(all_data['kitch_sq'].isnull() == True)]
all_data_kitch_sq_NaN_index = all_data['kitch_sq'].index[all_data['kitch_sq'].apply(np.isnan)]

train_data_kitch_sq_NaN = train_df['kitch_sq'].ix[(train_df['kitch_sq'].isnull() == True)]
train_data_kitch_sq_NaN_index = train_df['kitch_sq'].index[train_df['kitch_sq'].apply(np.isnan)]

test_data_kitch_sq_NaN = test_df['kitch_sq'].ix[(test_df['kitch_sq'].isnull() == True)]
test_data_kitch_sq_NaN_index = test_df['kitch_sq'].index[test_df['kitch_sq'].apply(np.isnan)]

all_data_school_quota_NaN = all_data['school_quota'].ix[(all_data['school_quota'].isnull() == True)]
all_data_school_quota_NaN_index = all_data['school_quota'].index[all_data['school_quota'].apply(np.isnan)]

train_data_school_quota_NaN = train_df['school_quota'].ix[(train_df['school_quota'].isnull() == True)]
train_data_school_quota_NaN_index = train_df['school_quota'].index[train_df['school_quota'].apply(np.isnan)]

test_data_school_quota_NaN = test_df['school_quota'].ix[(test_df['school_quota'].isnull() == True)]
test_data_school_quota_NaN_index = test_df['school_quota'].index[test_df['school_quota'].apply(np.isnan)]

all_data_preschool_quota_NaN = all_data['preschool_quota'].ix[(all_data['preschool_quota'].isnull() == True)]
all_data_preschool_quota_NaN_index = all_data['preschool_quota'].index[all_data['preschool_quota'].apply(np.isnan)]

train_data_preschool_quota_NaN = train_df['preschool_quota'].ix[(train_df['preschool_quota'].isnull() == True)]
train_data_preschool_quota_NaN_index = train_df['preschool_quota'].index[train_df['preschool_quota'].apply(np.isnan)]

test_data_preschool_quota_NaN = test_df['preschool_quota'].ix[(test_df['preschool_quota'].isnull() == True)]
test_data_preschool_quota_NaN_index = test_df['preschool_quota'].index[test_df['preschool_quota'].apply(np.isnan)]

all_data_max_floor_NaN = all_data['max_floor'].ix[(all_data['max_floor'].isnull() == True)]
all_data_max_floor_NaN_index = all_data['max_floor'].index[all_data['max_floor'].apply(np.isnan)]

train_data_max_floor_NaN = train_df['max_floor'].ix[(train_df['max_floor'].isnull() == True)]
train_data_max_floor_NaN_index = train_df['max_floor'].index[train_df['max_floor'].apply(np.isnan)]

test_data_max_floor_NaN = test_df['max_floor'].ix[(test_df['max_floor'].isnull() == True)]
test_data_max_floor_NaN_index = test_df['max_floor'].index[test_df['max_floor'].apply(np.isnan)]


all_data_full_sq_NaN = all_data['full_sq'].ix[(all_data['full_sq'].isnull() == True)]
all_data_full_sq_NaN_index = all_data['full_sq'].index[all_data['full_sq'].apply(np.isnan)]

train_data_full_sq_NaN = train_df['full_sq'].ix[(train_df['full_sq'].isnull() == True)]
train_data_full_sq_NaN_index = train_df['full_sq'].index[train_df['full_sq'].apply(np.isnan)]

test_data_full_sq_NaN = test_df['full_sq'].ix[(test_df['full_sq'].isnull() == True)]
test_data_full_sq_NaN_index = test_df['full_sq'].index[test_df['full_sq'].apply(np.isnan)]

all_data_build_year_NaN = all_data['build_year'].ix[(all_data['build_year'].isnull() == True)]
all_data_build_year_NaN_index = all_data['build_year'].index[all_data['build_year'].apply(np.isnan)]

all_data_build_year_NotNaN = all_data['build_year'].ix[(all_data['build_year'].isnull() == False)]
all_data_build_year_NotNaN_index = all_data['build_year'].index[(all_data['build_year'].isnull() == False)]

train_data_build_year_NaN = train_df['build_year'].ix[(train_df['build_year'].isnull() == True)]
train_data_build_year_NaN_index = train_df['build_year'].index[train_df['build_year'].apply(np.isnan)]

test_data_build_year_NaN = test_df['build_year'].ix[(test_df['build_year'].isnull() == True)]
test_data_build_year_NaN_index = test_df['build_year'].index[test_df['build_year'].apply(np.isnan)]

all_data_floor_NaN = all_data['floor'].ix[(all_data['floor'].isnull() == True)]
all_data_floor_NaN_index = all_data['floor'].index[all_data['floor'].apply(np.isnan)]

train_data_floor_NaN = train_df['floor'].ix[(train_df['floor'].isnull() == True)]
train_data_floor_NaN_index = train_df['floor'].index[train_df['floor'].apply(np.isnan)]

test_data_floor_NaN = test_df['floor'].ix[(test_df['floor'].isnull() == True)]
test_data_floor_NaN_index = test_df['floor'].index[test_df['floor'].apply(np.isnan)]

##***************************FIX floor: add median +- 1**********************************************************************************************************

print("ADDING MISSING floor VALUES...")

train_df['floor'].ix[(train_df['floor'].isnull() == True)] = train_df['floor'].ix[(train_df['floor'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['floor'].median()-1, all_data_without_NaN['floor'].median()+1))
test_df['floor'].ix[(test_df['floor'].isnull() == True)] = test_df['floor'].ix[(test_df['floor'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['floor'].median()-1, all_data_without_NaN['floor'].median()+1))

all_data['floor'].ix[(all_data['floor'].isnull() == True)] = all_data['floor'].ix[(all_data['floor'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['floor'].median()-1, all_data_without_NaN['floor'].median()+1))

print("ADDED MISSING floor VALUES")

##***************************FIX build_year with: 'floor','bulvar_ring_km','zd_vokzaly_avto_km','sadovoe_km','kremlin_km','ttk_km'**********************************************************************************************************

print("ADDING MISSING build_year VALUES...")

train_df_without_NaN = train_df.dropna()
test_df_without_NaN = test_df.dropna()
all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))

DataMostCorrelatedToBuildYear = all_data_without_NaN[['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km',
                                                     'kremlin_km', 'ttk_km']]

train_X = DataMostCorrelatedToBuildYear[['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km',
                                                     'kremlin_km', 'ttk_km']].values

train_y = all_data_without_NaN['build_year']

test_X = all_data.ix[(all_data['build_year'].isnull() == False)][['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km', 'kremlin_km', 'ttk_km']].values

build_year_test_X = all_data.ix[(all_data['build_year'].isnull() == False)][['build_year']].values

dtrain = xgb.DMatrix(train_X, label = train_y)

earlyStopRounds = 10
currentParams = {"max_depth":4, "gamma": 0.1,
                 "eta":0.01, "colsample_bytree":0.75,
                 "subsample": 0.75, 'objective': 'reg:linear',
                 'min_child_weight':1}
SEED = 0

##model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
##        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
##print("ERROR: ", model.values[model.shape[0]-1][0])
##print("BEST ROUND: ", model.shape[0])

best_num_boost_round = 4213
best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))

model_xgb = xgb.XGBRegressor(
    objective="reg:linear",
    n_estimators=best_num_boost_round,
    max_depth=4,
    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
    gamma=0.1)

model_xgb.fit(train_X, train_y)

y_pred = model_xgb.predict(test_X)
y_pred = np.around(y_pred)

meanDifference = np.mean(y_pred - build_year_test_X)

BuildYearsToPredict_Train = train_df.ix[(train_df['build_year'].isnull() == True)][['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km', 'kremlin_km', 'ttk_km']].values
PredictedMissingBuildYears_Train = model_xgb.predict(BuildYearsToPredict_Train)
PredictedMissingBuildYears_Train = np.around(PredictedMissingBuildYears_Train - meanDifference)

BuildYearsToPredict_Test = test_df.ix[(test_df['build_year'].isnull() == True)][['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km', 'kremlin_km', 'ttk_km']].values
PredictedMissingBuildYears_Test = model_xgb.predict(BuildYearsToPredict_Test)
PredictedMissingBuildYears_Test = np.around(PredictedMissingBuildYears_Test - meanDifference)



mean_build_year_before_train = train_df['build_year'].mean()
mean_build_year_before_test = test_df['build_year'].mean()

countOfPredictedValue_train = 0
for value in train_data_build_year_NaN_index:
    train_df.iloc[value,7] = PredictedMissingBuildYears_Train[countOfPredictedValue_train]
    countOfPredictedValue_train = countOfPredictedValue_train + 1

countOfPredictedValue_test = 0
for value in test_data_build_year_NaN_index:
    test_df.iloc[value,7] = PredictedMissingBuildYears_Test[countOfPredictedValue_test]
    countOfPredictedValue_test = countOfPredictedValue_test + 1

mean_build_year_after_train = train_df['build_year'].mean()
mean_build_year_after_test = test_df['build_year'].mean()

DifferenceOfMean_train = mean_build_year_before_train - mean_build_year_after_train
DifferenceOfMean_test = mean_build_year_before_test - mean_build_year_after_test

print("ADDED MISSING build_year VALUES")

##***************************Fix max_floor with: floor, build_year**********************************************************************************************************

print("ADDING MISSING max_floor VALUES...")

train_df_without_NaN = train_df.dropna()
test_df_without_NaN = test_df.dropna()
all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))

DataMostCorrelatedTo_Max_Floor = all_data_without_NaN[['floor', 'build_year']]

train_X = DataMostCorrelatedTo_Max_Floor.values
train_y = all_data_without_NaN['max_floor']
test_X = all_data.ix[(all_data['max_floor'].isnull() == False)][['floor', 'build_year']].values
max_floor_test_X = all_data.ix[(all_data['max_floor'].isnull() == False)][['max_floor']].values


dtrain = xgb.DMatrix(train_X, label = train_y)

earlyStopRounds = 10
currentParams = {"max_depth":4, "gamma": 0.1,
                 "eta":0.01, "colsample_bytree":0.75,
                 "subsample": 0.75, 'objective': 'reg:linear',
                 'min_child_weight':1}
SEED = 0

##model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
##        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
##print("ERROR: ", model.values[model.shape[0]-1][0])
##print("BEST ROUND: ", model.shape[0])

##ERROR:  3.0662888
##BEST ROUND:  761
best_num_boost_round = 761
best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))

model_xgb = xgb.XGBRegressor(
    objective="reg:linear",
    n_estimators=best_num_boost_round,
    max_depth=4,
    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
    gamma=0.1)

model_xgb.fit(train_X, train_y)

y_pred = model_xgb.predict(test_X)
y_pred = np.around(y_pred)

meanDifference = np.mean(y_pred - max_floor_test_X)

max_floor_ToPredict_Train = train_df.ix[(train_df['max_floor'].isnull() == True)][['floor', 'build_year']].values
PredictedMissing_max_floor_Train = model_xgb.predict(max_floor_ToPredict_Train)
PredictedMissing_max_floor_Train = np.around(PredictedMissing_max_floor_Train)

max_floor_ToPredict_Test = test_df.ix[(test_df['max_floor'].isnull() == True)][['floor', 'build_year']].values
PredictedMissing_max_floor_Test = model_xgb.predict(max_floor_ToPredict_Test)
PredictedMissing_max_floor_Test = np.around(PredictedMissing_max_floor_Test)


countOfPredictedValue_train = 0
for value in train_data_max_floor_NaN_index:
    train_df.iloc[value,5] = PredictedMissing_max_floor_Train[countOfPredictedValue_train]
    countOfPredictedValue_train = countOfPredictedValue_train + 1

countOfPredictedValue_test = 0
for value in test_data_max_floor_NaN_index:
    test_df.iloc[value,5] = PredictedMissing_max_floor_Test[countOfPredictedValue_test]
    countOfPredictedValue_test = countOfPredictedValue_test + 1

print("ADDED MISSING max_floor VALUES")

##***************************FIX full_sq: add median-+5**********************************************************************************************************

print("ADDING MISSING full_sq VALUES...")

train_df['full_sq'].ix[(train_df['full_sq'].isnull() == True)] = train_df['full_sq'].ix[(train_df['full_sq'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['full_sq'].median()-5, all_data_without_NaN['full_sq'].median()+5))

test_df['full_sq'].ix[(test_df['full_sq'].isnull() == True)] = test_df['full_sq'].ix[(test_df['full_sq'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['full_sq'].median()-5, all_data_without_NaN['full_sq'].median()+5))

print("ADDED MISSING full_sq VALUES")

##***************************FIX kitch_sq with: full_sq, max_floor(0.73), floor(0.38)**********************************************************************************************************

print("ADDING MISSING kitch_sq VALUES...")

train_df_without_NaN = train_df.dropna()
test_df_without_NaN = test_df.dropna()
all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))

DataMostCorrelatedTo_kitch_sq = all_data_without_NaN[['floor', 'max_floor', 'full_sq']]

train_X = DataMostCorrelatedTo_kitch_sq.values
##print(train_X.shape)##(7859, 3)
train_y = all_data_without_NaN['kitch_sq']
##print(train_y.values.shape)##((7859, 3)
test_X = all_data.ix[(all_data['kitch_sq'].isnull() == False)][['floor', 'max_floor', 'full_sq']].values
##print(test_X.shape)##(20088, 3)
kitch_sq_test_X = all_data.ix[(all_data['kitch_sq'].isnull() == False)][['kitch_sq']].values
##print(kitch_sq_test_X.shape)##(20088, 1)

dtrain = xgb.DMatrix(train_X, label = train_y)

earlyStopRounds = 10
currentParams = {"max_depth":4, "gamma": 0.1,
                 "eta":0.01, "colsample_bytree":0.75,
                 "subsample": 0.75, 'objective': 'reg:linear',
                 'min_child_weight':1}
SEED = 0

##model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
##        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
##print("ERROR: ", model.values[model.shape[0]-1][0])
##print("BEST ROUND: ", model.shape[0])

##ERROR:  1.8110122
##BEST ROUND:  410

best_num_boost_round = 475
best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))

model_xgb = xgb.XGBRegressor(
    objective="reg:linear",
    n_estimators=best_num_boost_round,
    max_depth=4,
    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
    gamma=0.1)

model_xgb.fit(train_X, train_y)

y_pred = model_xgb.predict(test_X)
y_pred = np.around(y_pred)

meanDifference = np.mean(y_pred - max_floor_test_X)
print(meanDifference)##-5.55576868502

kitch_sq_ToPredict_Train = train_df.ix[(train_df['kitch_sq'].isnull() == True)][['floor', 'max_floor', 'full_sq']].values
PredictedMissing_kitch_sq_Train = model_xgb.predict(kitch_sq_ToPredict_Train)
PredictedMissing_kitch_sq_Train = np.around(PredictedMissing_kitch_sq_Train - meanDifference)

kitch_sq_ToPredict_Test = test_df.ix[(test_df['kitch_sq'].isnull() == True)][['floor', 'max_floor', 'full_sq']].values
PredictedMissing_kitch_sq_Test = model_xgb.predict(kitch_sq_ToPredict_Test)
PredictedMissing_kitch_sq_Test = np.around(PredictedMissing_kitch_sq_Test - meanDifference)

mean_kitch_sq_before_train = train_df['kitch_sq'].mean()
mean_kitch_sq_before_test = test_df['kitch_sq'].mean()

print(train_df['kitch_sq'].describe())
print(test_df['kitch_sq'].describe())

countOfPredictedValue_train = 0
for value in train_data_kitch_sq_NaN_index:
##    print(value)
    train_df.iloc[value,9] = PredictedMissing_kitch_sq_Train[countOfPredictedValue_train] 
    countOfPredictedValue_train = countOfPredictedValue_train + 1

countOfPredictedValue_test = 0
for value in test_data_kitch_sq_NaN_index:
##    print(value)
    test_df.iloc[value,9] = PredictedMissing_kitch_sq_Test[countOfPredictedValue_test]
    countOfPredictedValue_test = countOfPredictedValue_test + 1

print(train_df['kitch_sq'].describe())
print(test_df['kitch_sq'].describe())

mean_kitch_sq_after_train = train_df['kitch_sq'].mean()
mean_kitch_sq_after_test = test_df['kitch_sq'].mean()

DifferenceOfMean_train = mean_kitch_sq_before_train - mean_kitch_sq_after_train
DifferenceOfMean_test = mean_kitch_sq_before_test - mean_kitch_sq_after_test

print(DifferenceOfMean_train)##-3.43280815146
print(DifferenceOfMean_test)##--1.73523629161

threshold = -1

while DifferenceOfMean_train < threshold:
    print('Difference too high for train data, need to add more!')
    countOfPredictedValue_train = 0
    for value in train_data_kitch_sq_NaN_index:
    ##    print(value)
        train_df.iloc[value,9] = train_df.iloc[value,9] + DifferenceOfMean_train
        countOfPredictedValue_train = countOfPredictedValue_train + 1
    mean_kitch_sq_after_train = train_df['kitch_sq'].mean()
    DifferenceOfMean_train = mean_kitch_sq_before_train - mean_kitch_sq_after_train

while DifferenceOfMean_test < threshold:
    print('Difference too high for test data, need to add more!')

    countOfPredictedValue_test = 0
    for value in test_data_kitch_sq_NaN_index:
    ##    print(value)
        test_df.iloc[value,9] = test_df.iloc[value,9] + DifferenceOfMean_test
        countOfPredictedValue_test = countOfPredictedValue_test + 1
    mean_kitch_sq_after_test = test_df['kitch_sq'].mean()
    DifferenceOfMean_test = mean_kitch_sq_before_test - mean_kitch_sq_after_test
    
##    
print("After adding difference of mean: ")
print(train_df['kitch_sq'].describe())
print(test_df['kitch_sq'].describe())


print("ADDED MISSING kitch_sq VALUES")

##***************************FIX num_room with: full_sq, kitch_sq**********************************************************************************************************

print("ADDING MISSING num_room VALUES...")

train_df_without_NaN = train_df.dropna()
test_df_without_NaN = test_df.dropna()
all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))

DataMostCorrelatedTo_num_room = all_data_without_NaN[['full_sq', 'kitch_sq']]

train_X = DataMostCorrelatedTo_num_room.values
##print(train_X.shape)##(7859, 3)
train_y = all_data_without_NaN['num_room']
##print(train_y.values.shape)##((7859, 3)
test_X = all_data.ix[(all_data['num_room'].isnull() == False)][['full_sq', 'kitch_sq']].values
##print(test_X.shape)##(20088, 3)
num_room_test_X = all_data.ix[(all_data['num_room'].isnull() == False)][['num_room']].values
##print(kitch_sq_test_X.shape)##(20088, 1)

dtrain = xgb.DMatrix(train_X, label = train_y)

earlyStopRounds = 10
currentParams = {"max_depth":4, "gamma": 0.1,
                 "eta":0.01, "colsample_bytree":0.75,
                 "subsample": 0.75, 'objective': 'reg:linear',
                 'min_child_weight':1}
SEED = 0

##model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
##        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
##print("ERROR: ", model.values[model.shape[0]-1][0])
##print("BEST ROUND: ", model.shape[0])

##ERROR:  0.4173778
##BEST ROUND:  1216

best_num_boost_round = 1216
best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))

model_xgb = xgb.XGBRegressor(
    objective="reg:linear",
    n_estimators=best_num_boost_round,
    max_depth=4,
    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
    gamma=0.1)

model_xgb.fit(train_X, train_y)

y_pred = model_xgb.predict(test_X)
y_pred = np.around(y_pred)

meanDifference = np.mean(y_pred - num_room_test_X)
print(meanDifference)##-11.5160221806

num_room_ToPredict_Train = train_df.ix[(train_df['num_room'].isnull() == True)][['full_sq', 'kitch_sq']].values
PredictedMissing_num_room_Train = model_xgb.predict(num_room_ToPredict_Train)
PredictedMissing_num_room_Train = np.around(PredictedMissing_num_room_Train)

num_room_ToPredict_Test = test_df.ix[(test_df['num_room'].isnull() == True)][['full_sq', 'kitch_sq']].values
PredictedMissing_num_room_Test = model_xgb.predict(num_room_ToPredict_Test)
PredictedMissing_num_room_Test = np.around(PredictedMissing_num_room_Test)

mean_num_room_before_train = train_df['num_room'].mean()
mean_num_room_before_test = test_df['num_room'].mean()

print(train_df['num_room'].describe())
print(test_df['num_room'].describe())

countOfPredictedValue_train = 0
for value in train_data_num_room_NaN_index:
##    print(value)
    train_df.iloc[value,8] = PredictedMissing_num_room_Train[countOfPredictedValue_train] 
    countOfPredictedValue_train = countOfPredictedValue_train + 1

countOfPredictedValue_test = 0
for value in test_data_num_room_NaN_index:
##    print(value)
    test_df.iloc[value,8] = PredictedMissing_num_room_Test[countOfPredictedValue_test]
    countOfPredictedValue_test = countOfPredictedValue_test + 1

print(train_df['num_room'].describe())
print(test_df['num_room'].describe())

mean_num_room_after_train = train_df['num_room'].mean()
mean_num_room_after_test = test_df['num_room'].mean()

DifferenceOfMean_train = mean_num_room_before_train - mean_num_room_after_train
DifferenceOfMean_test = mean_num_room_before_test - mean_num_room_after_test

##print(DifferenceOfMean_train)##0.117363361075
##print(DifferenceOfMean_test)##0.000228077811298


print("ADDED MISSING num_room VALUES")

##***************************FIX life_sq with: full_sq, num_room**********************************************************************************************************

print("ADDING MISSING life_sq VALUES...")

train_df_without_NaN = train_df.dropna()
test_df_without_NaN = test_df.dropna()
all_data_without_NaN = pd.concat((train_df_without_NaN.loc[:,'timestamp':'market_count_5000'],
                      test_df_without_NaN.loc[:,'timestamp':'market_count_5000']))


DataMostCorrelatedTo_life_sq = all_data_without_NaN[['full_sq', 'num_room']]

train_X = DataMostCorrelatedTo_life_sq.values
##print(train_X.shape)##(7859, 3)
train_y = all_data_without_NaN['life_sq']
##print(train_y.values.shape)##((7859, 3)
test_X = all_data.ix[(all_data['life_sq'].isnull() == False)][['full_sq', 'num_room']].values
##print(test_X.shape)##(20088, 3)
life_sq_test_X = all_data.ix[(all_data['life_sq'].isnull() == False)][['life_sq']].values
##print(kitch_sq_test_X.shape)##(20088, 1)

dtrain = xgb.DMatrix(train_X, label = train_y)

earlyStopRounds = 10
currentParams = {"max_depth":4, "gamma": 0.1,
                 "eta":0.01, "colsample_bytree":0.75,
                 "subsample": 0.75, 'objective': 'reg:linear',
                 'min_child_weight':1}
SEED = 0

##model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
##        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
##print("ERROR: ", model.values[model.shape[0]-1][0])
##print("BEST ROUND: ", model.shape[0])

##ERROR:  7.9550282
##BEST ROUND:  617

best_num_boost_round = 617
best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))

model_xgb = xgb.XGBRegressor(
    objective="reg:linear",
    n_estimators=best_num_boost_round,
    max_depth=4,
    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
    gamma=0.1)

model_xgb.fit(train_X, train_y)

y_pred = model_xgb.predict(test_X)
y_pred = np.around(y_pred)

meanDifference = np.mean(y_pred - life_sq_test_X)
print(meanDifference)##-3.02642641129

life_sq_ToPredict_Train = train_df.ix[(train_df['life_sq'].isnull() == True)][['full_sq', 'num_room']].values
PredictedMissing_life_sq_Train = model_xgb.predict(life_sq_ToPredict_Train)
PredictedMissing_life_sq_Train = np.around(PredictedMissing_life_sq_Train)

life_sq_ToPredict_Test = test_df.ix[(test_df['life_sq'].isnull() == True)][['full_sq', 'num_room']].values
PredictedMissing_life_sq_Test = model_xgb.predict(life_sq_ToPredict_Test)
PredictedMissing_life_sq_Test = np.around(PredictedMissing_life_sq_Test)





mean_life_sq_before_train = train_df['life_sq'].mean()
mean_life_sq_before_test = test_df['life_sq'].mean()

print(train_df['life_sq'].describe())
print(test_df['life_sq'].describe())

countOfPredictedValue_train = 0
for value in train_data_life_sq_NaN_index:
##    print(value)
    train_df.iloc[value,3] = PredictedMissing_life_sq_Train[countOfPredictedValue_train] 
    countOfPredictedValue_train = countOfPredictedValue_train + 1

countOfPredictedValue_test = 0
for value in test_data_life_sq_NaN_index:
##    print(value)
    test_df.iloc[value,3] = PredictedMissing_life_sq_Test[countOfPredictedValue_test]
    countOfPredictedValue_test = countOfPredictedValue_test + 1

print(train_df['life_sq'].describe())
print(test_df['life_sq'].describe())

mean_life_sq_after_train = train_df['life_sq'].mean()
mean_life_sq_after_test = test_df['life_sq'].mean()

DifferenceOfMean_train = mean_life_sq_before_train - mean_life_sq_after_train
DifferenceOfMean_test = mean_life_sq_before_test - mean_life_sq_after_test

print(DifferenceOfMean_train)##0.0539532671414
print(DifferenceOfMean_test)##0.335228447754
##
##print("ADDED MISSING life_sq VALUES")


##***************************Give everything a random for now(20/06/17, 6:15)**********************************************************************************************************

##train_df['life_sq'].ix[(train_df['life_sq'].isnull() == True)] = train_df['life_sq'].ix[(train_df['life_sq'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['life_sq'].median()-3), int(all_data_without_NaN['life_sq'].median()+3)))
##test_df['life_sq'].ix[(test_df['life_sq'].isnull() == True)] = test_df['life_sq'].ix[(test_df['life_sq'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['life_sq'].median()-3), int(all_data_without_NaN['life_sq'].median()+3)))
##  
##train_df['kitch_sq'].ix[(train_df['kitch_sq'].isnull() == True)] = train_df['kitch_sq'].ix[(train_df['kitch_sq'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['kitch_sq'].median()-3), int(all_data_without_NaN['kitch_sq'].median()+3)))
##test_df['kitch_sq'].ix[(test_df['kitch_sq'].isnull() == True)] = test_df['kitch_sq'].ix[(test_df['kitch_sq'].isnull() == True)].apply(lambda v: randint(int(all_data_without_NaN['kitch_sq'].median()-3), int(all_data_without_NaN['kitch_sq'].median()+3)))
##
##train_df['num_room'].ix[(train_df['num_room'].isnull() == True)] = train_df['num_room'].ix[(train_df['num_room'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['num_room'].median()-1, all_data_without_NaN['num_room'].median()+1))
##test_df['num_room'].ix[(test_df['num_room'].isnull() == True)] = test_df['num_room'].ix[(test_df['num_room'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['num_room'].median()-1, all_data_without_NaN['num_room'].median()+1))
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
##train_df['hospital_beds_raion'].ix[(train_df['hospital_beds_raion'].isnull() == True)] = train_df['hospital_beds_raion'].ix[(train_df['hospital_beds_raion'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['hospital_beds_raion'].median()-500, all_data_without_NaN['hospital_beds_raion'].median()+500))
##test_df['hospital_beds_raion'].ix[(test_df['hospital_beds_raion'].isnull() == True)] = test_df['hospital_beds_raion'].ix[(test_df['hospital_beds_raion'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['hospital_beds_raion'].median()-500, all_data_without_NaN['hospital_beds_raion'].median()+500))
##
##train_df['state'].ix[(train_df['state'].isnull() == True)] = train_df['state'].ix[(train_df['state'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['state'].median()-1, all_data_without_NaN['state'].median()+1))
##test_df['state'].ix[(test_df['state'].isnull() == True)] = test_df['state'].ix[(test_df['state'].isnull() == True)].apply(lambda v: randint(all_data_without_NaN['state'].median()-1, all_data_without_NaN['state'].median()+1))
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
##
######***************************Predict school_quota with: raion_popul,work_female,work_all,0_17_female,children_preschool,0_6_all,0_17_all,0_6_male,0_13_all**********************************************************************************************************
##
##DataMostCorrelatedTo_school_quota = all_data_without_NaN[['raion_popul','work_female','work_all','0_17_female','children_preschool','0_6_all','0_17_all','0_6_male','0_13_all']]
##train_X = DataMostCorrelatedTo_school_quota.values
##print(train_X.shape)##(7859, 8)
##train_y = all_data_without_NaN['school_quota']
##print(train_y.values.shape)##(7859,)
##test_X = all_data.ix[(all_data['school_quota'].isnull() == False)][['raion_popul','work_female','work_all','0_17_female','children_preschool','0_6_all','0_17_all','0_6_male','0_13_all']].values
##print(test_X.shape)##(29853, 9)
##school_quota_test_X = all_data.ix[(all_data['school_quota'].isnull() == False)][['school_quota']].values
##print(school_quota_test_X.shape)##(29853, 1)
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
##print(meanDifference)##-908.972398084
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
##print(train_df['school_quota'].describe())
##print(test_df['school_quota'].describe())
####
##mean_school_quota_before_train = train_df['school_quota'].mean()
##mean_school_quota_before_test = test_df['school_quota'].mean()
##
####print(mean_school_quota_before_train)
####print(mean_school_quota_before_test)
##
##print("ADDING MISSING school_quota VALUES...")
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
##print("Difference of Mean train: ", DifferenceOfMean_train)
##print("Difference of Mean test: ", DifferenceOfMean_test)
##
##while DifferenceOfMean_train > 100:
##    print('Difference too high for train data, need to add more!')
##    countOfPredictedValue_train = 0
##    for value in train_data_school_quota_NaN_index:
##    ##    print(value)
##        train_df.iloc[value,21] = train_df.iloc[value,21] + DifferenceOfMean_train
##        countOfPredictedValue_train = countOfPredictedValue_train + 1
##    mean_school_quota_after_train = train_df['school_quota'].mean()
##    DifferenceOfMean_train = mean_school_quota_before_train - mean_school_quota_after_train
##
##while DifferenceOfMean_test > 100:
##    print('Difference too high for test data, need to add more!')
##
##    countOfPredictedValue_test = 0
##    for value in test_data_school_quota_NaN_index:
##    ##    print(value)
##        test_df.iloc[value,21] = test_df.iloc[value,21] + DifferenceOfMean_test
##        countOfPredictedValue_test = countOfPredictedValue_test + 1
##    mean_school_quota_after_test = test_df['school_quota'].mean()
##    DifferenceOfMean_test = mean_school_quota_before_test - mean_school_quota_after_test
##    
##print("After adding difference of mean: ")
##print(train_df['school_quota'].describe())
##print(test_df['school_quota'].describe())
##
##
######***************************Predict preschool_quota with: work_female, raion_popul, work_all, work_male, children_preschool, 0_6_all, 0_6_male**********************************************************************************************************
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
##print(meanDifference)##-432.33880532
##
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
##print(train_df['preschool_quota'].describe())
##print(test_df['preschool_quota'].describe())
##
##mean_preschool_quota_before_train = train_df['preschool_quota'].mean()
##mean_preschool_quota_before_test = test_df['preschool_quota'].mean()
##
##print("ADDING MISSING preschool_quota VALUES...")
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
##print("Difference of Mean train: ", DifferenceOfMean_train)
##print("Difference of Mean test: ", DifferenceOfMean_test)
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
##    print('Difference too high for train data, need to add more!')
##    countOfPredictedValue_train = 0
##    for value in train_data_preschool_quota_NaN_index:
##    ##    print(value)
##        train_df.iloc[value,18] = train_df.iloc[value,18] + DifferenceOfMean_train
##        countOfPredictedValue_train = countOfPredictedValue_train + 1
##    mean_preschool_quota_after_train = train_df['preschool_quota'].mean()
##    DifferenceOfMean_train = mean_preschool_quota_before_train - mean_preschool_quota_after_train
##
##while DifferenceOfMean_test > 100:
##    print('Difference too high for test data, need to add more!')
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
##print(train_df['preschool_quota'].describe())
##print(test_df['preschool_quota'].describe())

























##***************************BELOW ARE PRINTS*********************************************************************************************************


##print(train_df_without_NaN.values.shape)##(5662, 292)
##print(test_df_without_NaN.values.shape)##(2197, 291)
##print(all_data_without_NaN.values.shape)##(7859, 290)

##print(all_data['build_year'][2000])##<----IT RETURNS: 2000 NaN, 2000 2008.0, 2 records!

##print(all_data.values.shape)##(38133, 290)
##
##print(all_data_build_year_NaN.values.shape)##(16110,)
##print(all_data_build_year_NaN_index)##length=16110
##
##print(all_data_build_year_NotNaN.values.shape)##(22023,)
##print(all_data_build_year_NotNaN_index)##length=22023

##print(train_data_build_year_NaN.values.shape)##(14505,)
##print(train_data_build_year_NaN_index)##length=14505

##print(test_data_build_year_NaN.values.shape)##(1605,)
##print(test_data_build_year_NaN_index)##length=1605

##print(all_data_floor_NaN.values.shape)##(177,)
##print(all_data_floor_NaN_index)##length=177

##print(train_data_floor_NaN.values.shape)##(177,)
##print(train_data_floor_NaN_index)##length=177
##
##print(test_data_floor_NaN.values.shape)##(0,)
##print(test_data_floor_NaN_index)##length=0

