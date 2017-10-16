import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import skew

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

import random
from sklearn.decomposition import PCA

df_train = pd.read_csv('train.csv')
##print(all_data.columns)
df_test = pd.read_csv('test.csv')

all_data = pd.concat((df_train.loc[:,'timestamp':'market_count_5000'],
                      df_test.loc[:,'timestamp':'market_count_5000']))


##print(all_data['product_type'].mode())##0    Investment
all_data['product_type'] = all_data['product_type'].fillna('Investment')
all_data['green_part_2000'] = np.round(all_data['green_part_2000'].fillna(all_data['green_part_2000'].mean()))

all_data['hospital_beds_raion'] = np.round(all_data['hospital_beds_raion'].fillna(all_data['hospital_beds_raion'].mean()))
all_data = all_data.drop('build_year',1)
all_data['state'] = np.round(all_data['state'].fillna(all_data['state'].median()))
##all_data = all_data.drop('cafe_avg_price_500',1)
##all_data = all_data.drop('cafe_sum_500_max_price_avg',1)
##all_data = all_data.drop('cafe_sum_500_min_price_avg',1)
all_data = all_data.drop('max_floor',1)
all_data = all_data.drop('material',1)
all_data['num_room'] = np.round(all_data['num_room'].fillna(all_data['num_room'].median()))
all_data['kitch_sq'] = np.round(all_data['kitch_sq'].fillna(all_data['kitch_sq'].mean()))
all_data['preschool_quota'] = np.round(all_data['preschool_quota'].fillna(all_data['preschool_quota'].mean()))
all_data['school_quota'] = np.round(all_data['school_quota'].fillna(all_data['school_quota'].mean()))

all_data['life_sq'] = np.round(all_data['life_sq'].fillna(all_data['life_sq'].mean()))
all_data = all_data.drop('build_count_frame',1)
all_data = all_data.drop('build_count_1971-1995',1)
all_data = all_data.drop('build_count_block',1)
all_data = all_data.drop('build_count_wood',1)
all_data = all_data.drop('build_count_mix',1)
all_data = all_data.drop('build_count_1921-1945',1)
all_data = all_data.drop('build_count_panel',1)
all_data = all_data.drop('build_count_foam',1)
all_data = all_data.drop('build_count_slag',1)
all_data = all_data.drop('raion_build_count_with_builddate_info',1)
all_data = all_data.drop('build_count_monolith',1)
all_data = all_data.drop('build_count_before_1920',1)
all_data = all_data.drop('build_count_1946-1970',1)
all_data = all_data.drop('build_count_brick',1)
all_data = all_data.drop('raion_build_count_with_material_info',1)
all_data = all_data.drop('build_count_after_1995',1)

all_data = all_data.drop('cafe_avg_price_500',1)
all_data = all_data.drop('cafe_sum_500_min_price_avg',1)
all_data = all_data.drop('cafe_sum_500_max_price_avg',1)

all_data = all_data.drop('cafe_avg_price_1000',1)
all_data = all_data.drop('cafe_sum_1000_max_price_avg',1)
all_data = all_data.drop('cafe_sum_1000_min_price_avg',1)

all_data = all_data.drop('cafe_avg_price_1500',1)
all_data = all_data.drop('cafe_sum_1500_min_price_avg',1)
all_data = all_data.drop('cafe_sum_1500_max_price_avg',1)


all_data['cafe_avg_price_2000'] = all_data['cafe_avg_price_2000'].fillna(all_data['cafe_avg_price_2000'].median())
all_data['cafe_sum_2000_min_price_avg'] = all_data['cafe_sum_2000_min_price_avg'].fillna(all_data['cafe_sum_2000_min_price_avg'].median())
all_data['cafe_sum_2000_max_price_avg'] = all_data['cafe_sum_2000_max_price_avg'].fillna(all_data['cafe_sum_2000_max_price_avg'].median())

##all_data = all_data.drop('cafe_avg_price_2000',1)
##all_data = all_data.drop('cafe_sum_2000_max_price_avg',1)
##all_data = all_data.drop('cafe_sum_2000_min_price_avg',1)

all_data['cafe_avg_price_3000'] = all_data['cafe_avg_price_3000'].fillna(all_data['cafe_avg_price_3000'].median())
all_data['cafe_sum_3000_min_price_avg'] = all_data['cafe_sum_3000_min_price_avg'].fillna(all_data['cafe_sum_3000_min_price_avg'].median())
all_data['cafe_sum_3000_max_price_avg'] = all_data['cafe_sum_3000_max_price_avg'].fillna(all_data['cafe_sum_3000_max_price_avg'].median())

##
##all_data = all_data.drop('cafe_avg_price_3000',1)
##all_data = all_data.drop('cafe_sum_3000_max_price_avg',1)
##all_data = all_data.drop('cafe_sum_3000_min_price_avg',1)
##

all_data['cafe_avg_price_5000'] = all_data['cafe_avg_price_5000'].fillna(all_data['cafe_avg_price_5000'].median())
all_data['cafe_sum_5000_min_price_avg'] = all_data['cafe_sum_5000_min_price_avg'].fillna(all_data['cafe_sum_5000_min_price_avg'].median())
all_data['cafe_sum_5000_max_price_avg'] = all_data['cafe_sum_5000_max_price_avg'].fillna(all_data['cafe_sum_5000_max_price_avg'].median())

##all_data = all_data.drop('cafe_avg_price_5000',1)
##all_data = all_data.drop('cafe_sum_5000_max_price_avg',1)
##all_data = all_data.drop('cafe_sum_5000_min_price_avg',1)

all_data['prom_part_5000'] = all_data['prom_part_5000'].fillna(all_data['prom_part_5000'].mean())
all_data['floor'] = np.round(all_data['floor'].fillna(all_data['floor'].median()))
all_data['metro_min_walk'] = np.round(all_data['metro_min_walk'].fillna(all_data['metro_min_walk'].mean()))
all_data['railroad_station_walk_km'] = np.round(all_data['railroad_station_walk_km'].fillna(all_data['railroad_station_walk_km'].mean()))
all_data['railroad_station_walk_min'] = np.round(all_data['railroad_station_walk_min'].fillna(all_data['railroad_station_walk_min'].mean()))
all_data = all_data.drop('ID_railroad_station_walk',1)
all_data['metro_km_walk'] = np.round(all_data['metro_km_walk'].fillna(all_data['metro_km_walk'].mean()))

##DROP THE COLUMNS THAT DO NOT HAVE na BUT NOT RELEVANT TO OUR ANALYSIS
all_data = all_data.drop('green_zone_part',1)
all_data = all_data.drop('indust_part',1)


all_data = all_data.drop('thermal_power_plant_raion',1)
all_data = all_data.drop('incineration_raion',1)
all_data = all_data.drop('oil_chemistry_raion',1)
all_data = all_data.drop('radiation_raion',1)
all_data = all_data.drop('railroad_terminal_raion',1)
all_data = all_data.drop('big_market_raion',1)
all_data = all_data.drop('nuclear_reactor_raion',1)
all_data = all_data.drop('detention_facility_raion',1)



all_data = all_data.drop('full_all',1)
all_data = all_data.drop('male_f',1)
all_data = all_data.drop('female_f',1)
all_data = all_data.drop('young_all',1)
all_data = all_data.drop('work_all',1)
all_data = all_data.drop('ekder_all',1)
all_data = all_data.drop('0_6_all',1)
all_data = all_data.drop('7_14_all',1)
all_data = all_data.drop('0_17_all',1)
all_data = all_data.drop('0_17_male',1)
all_data = all_data.drop('0_17_female',1)
all_data = all_data.drop('16_29_all',1)
all_data = all_data.drop('0_13_all',1)
all_data = all_data.drop('0_13_male',1)
all_data = all_data.drop('0_13_female',1)

all_data = all_data.drop('metro_min_avto',1)
all_data = all_data.drop('metro_min_walk',1)
all_data = all_data.drop('railroad_station_walk_min',1)
all_data = all_data.drop('railroad_station_avto_min',1)
all_data = all_data.drop('ID_railroad_station_avto',1)
all_data = all_data.drop('public_transport_station_min_walk',1)
all_data = all_data.drop('water_1line',1)
all_data = all_data.drop('ID_big_road1',1)
all_data = all_data.drop('ID_big_road2',1)
all_data = all_data.drop('ID_railroad_terminal',1)
all_data = all_data.drop('ID_bus_terminal',1)

all_data = all_data.drop('ice_rink_km',1)
all_data = all_data.drop('basketball_km',1)
all_data = all_data.drop('swim_pool_km',1)
all_data = all_data.drop('detention_facility_km',1)
all_data = all_data.drop('ecology',1)

for col in all_data.columns:
    if 'cafe_count_500_na' in col:
        del all_data[col]

for col in all_data.columns:
    if 'cafe_count_1000_na' in col:
        del all_data[col]

for col in all_data.columns:
    if 'cafe_count_1500_na' in col:
        del all_data[col]

for col in all_data.columns:
    if 'cafe_count_2000_' in col:
        del all_data[col]        

for col in all_data.columns:
    if 'cafe_count_3000_' in col:
        del all_data[col]        

for col in all_data.columns:
    if 'cafe_count_5000_' in col:
        del all_data[col]

##AFTER DROPPING THE MISSING ONES/FILLING NA, WHAT I WANT TO DO:
##1. Use PCA to reduce data dimensionality, possibly 100
##2. Use StandardScaler to standardize remaining FEATURES by removing the mean and scaling to unit variance

##We cannot use StandardScaler for price_doc, so just log1p         
df_train['price_doc'] = np.log1p(df_train['price_doc'])
y = df_train.price_doc


all_data = pd.get_dummies(all_data)
##print(all_data.values.shape)##(38133, 1779)
##print(all_data)
pca = PCA(n_components=100)
all_data = pca.fit_transform(all_data)


all_data = StandardScaler().fit_transform(all_data)



X_train = all_data[:df_train.shape[0]]
##print(X_train.shape)
X_test = all_data[df_train.shape[0]:]
##print(X_test.shape)

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)


LowestError = 10000
params = {"max_depth":3, "eta":1, "alpha": 1, "colsample_bytree":0.5, "subsample": 0.5 }
bestParams = params
bestBoostRound = 100
bestCVLength = 100000

for currentBoostRound in np.arange(2000, 2010, 10):##1000, 10010, 10, got to roughly 660 on 8/06/2017, 7:10
    for currentEta in np.arange(0.01, 0.31, 0.1):
        for currentColsampleByTree in np.arange(0.5,1.0,0.1):
            for currentSubsample in np.arange(0.5,1.0,0.1):
                for currentGamma in np.arange(0,1.1,0.1):
                    for currentDep in np.arange(3, 16, 1):##3, 16, 1
                        currentParams = {"max_depth":currentDep, "gamma": currentGamma ,"eta":currentEta, "alpha": 1, "colsample_bytree":currentColsampleByTree, "subsample": currentSubsample }
                        print("NEW CROSS VALIDATION RUNNING....")
                        print("currentBoostRound: ", currentBoostRound)
                        print("currentEta: ",currentEta)
                        print("currentColsampleByTree: ", currentColsampleByTree)
                        print("currentSubsample: ", currentSubsample)
                        print("currentGamma: ", currentGamma)
                        print("currentDep: ", currentDep)
                        print("currentParams: ",currentParams)

                        model = xgb.cv(currentParams, dtrain,  num_boost_round=currentBoostRound, early_stopping_rounds=50, nfold = 5, verbose_eval=5)
                        

                        if model.values[model.shape[0]-1][0] < LowestError:
                            LowestError = model.values[model.shape[0]-1][0]
                            bestBoostRound = currentBoostRound
                            bestParams = currentParams
                            bestCVLength = model.shape[0]
                        print("LowestError: ", LowestError)
                        print("Best BoostRound: ", bestBoostRound)
                        print("Best Parameter: ", bestParams )
                        print("Best CVLength: ", bestCVLength )
                        print(model.values[model.shape[0]-1])
                        f = open('XGBTuningRecordPCA_SS.txt', 'w')
                        f.write("Best so far: ")
                        f.write('\n')
                        f.write('\n')
                        f.write(str(LowestError))
                        f.write('\n')
                        f.write(str(bestBoostRound))
                        f.write('\n')
                        f.write(str(bestParams))
                        f.write('\n')
                        f.write(str(bestCVLength))
                        f.write('\n')
                        f.write('\n')
                        f.write("Last tested set of parameters and BoostRound: ")
                        f.write('\n')
                        f.write('\n')
                        f.write(str(currentBoostRound))
                        f.write('\n')
                        f.write(str(currentParams))
                        f.write('\n')
                        f.write(str(model.shape[0]))                        
                        f.close()
                        print("Written to file, go to next iteration")
                        print('\n')



##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=1000,
##    max_depth=5,
##    learning_rate=0.05,reg_alpha=1,subsample=0.7, colsample_bytree=0.7, gamma=0.0385) #the params were tuned using xgb.cv
##
##model_xgb.fit(X_train, y)
##xgb_preds = np.expm1(model_xgb.predict(X_test))
##xgb_preds = np.around(xgb_preds)
##preds = xgb_preds
##solution = pd.DataFrame({"id":df_test.id, "price_doc":preds})
##solution.to_csv("RussianHousingPredict.csv", index = False)
                     

print("END OF PROGRAM")

        
