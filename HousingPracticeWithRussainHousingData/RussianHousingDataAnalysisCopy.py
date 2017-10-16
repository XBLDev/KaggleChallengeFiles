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

df_train = pd.read_csv('train.csv')
##print(all_data.columns)
df_test = pd.read_csv('test.csv')

all_data = pd.concat((df_train.loc[:,'timestamp':'market_count_5000'],
                      df_test.loc[:,'timestamp':'market_count_5000']))

##print(all_data.describe())


##print(all_data['price_doc'].describe())
##sns.distplot(all_data['price_doc']);
##sns.plt.show()

#skewness and kurtosis
##print("Skewness: %f" % all_data['price_doc'].skew())
##print("Kurtosis: %f" % all_data['price_doc'].kurt())


#scatter plot grlivarea/saleprice
##var = 'life_sq'##life_sq
##data = pd.concat([all_data['price_doc'], all_data[var]], axis=1)
##data.plot.scatter(x=var, y='price_doc', ylim=(0,10000000), xlim=(0,200));
##plt.show()


#box plot overallqual/saleprice
##var = 'build_year'##ecology
##data = pd.concat([all_data['price_doc'], all_data[var]], axis=1)
##f, ax = plt.subplots(figsize=(8, 6))
##fig = sns.boxplot(x=var, y="price_doc", data=data)
##fig.axis(ymin=0, ymax=10000000);
##plt.xticks(rotation=90)
##sns.plt.show()

##var = 'sub_area'##sub_area
##data = pd.concat([all_data['price_doc'], all_data[var]], axis=1)
##f, ax = plt.subplots(figsize=(8, 6))
##fig = sns.boxplot(x=var, y="price_doc", data=data)
##fig.axis(ymin=0, ymax=10000000);
##sns.plt.show()

##var = 'num_room'##sub_area
##data = pd.concat([all_data['price_doc'], all_data[var]], axis=1)
##data.plot.scatter(x=var, y='price_doc', ylim=(0,10000000), xlim=(0,5));
##plt.show()

#correlation matrix
##corrmat = all_data.corr()
##f, ax = plt.subplots(figsize=(12, 9))
##sns.heatmap(corrmat, vmax=.8, square=True);
##sns.plt.show()

##corrmat = all_data.corr()
####saleprice correlation matrix
##k = 10 #number of variables for heatmap
##cols = corrmat.nlargest(k, 'price_doc')['price_doc'].index
####cols = corrmat.nlargest(k, 'kitch_sq')['kitch_sq'].index
##cm = np.corrcoef(all_data[cols].values.T)
##sns.set(font_scale=1.25)
##hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
##plt.yticks(rotation=0)
##plt.xticks(rotation=90)
##plt.show()
##sns.plt.show()

#missing data
##total = all_data.isnull().sum().sort_values(ascending=False)
##percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
##missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
##print(missing_data[missing_data['Total'] > 1])
##print(missing_data)
##We'll consider that when more than 15% of the data is missing, we should delete the corresponding variable and pretend it never existed. 
##print(missing_data.head(20))

##*********************MISSING DATA TOP*****************
##                                       Total   Percent
##hospital_beds_raion                    14441  0.473926
##build_year                             13605  0.446490
##state                                  13559  0.444980
##cafe_avg_price_500                     13281  0.435857
##cafe_sum_500_max_price_avg             13281  0.435857
##cafe_sum_500_min_price_avg             13281  0.435857
##max_floor                               9572  0.314135
##material                                9572  0.314135
##num_room                                9572  0.314135
##kitch_sq                                9572  0.314135
##preschool_quota                         6688  0.219487
##school_quota                            6685  0.219389
##cafe_sum_1000_min_price_avg             6524  0.214105
##cafe_sum_1000_max_price_avg             6524  0.214105
##cafe_avg_price_1000                     6524  0.214105
##life_sq                                 6383  0.209478
##build_count_frame                       4991  0.163795
##build_count_1971-1995                   4991  0.163795
##build_count_block                       4991  0.163795
##raion_build_count_with_material_info    4991  0.163795
##build_count_after_1995                  4991  0.163795
##build_count_brick                       4991  0.163795
##build_count_wood                        4991  0.163795
##build_count_mix                         4991  0.163795
##build_count_1921-1945                   4991  0.163795
##build_count_panel                       4991  0.163795
##build_count_foam                        4991  0.163795
##build_count_slag                        4991  0.163795
##raion_build_count_with_builddate_info   4991  0.163795
##build_count_monolith                    4991  0.163795
##build_count_before_1920                 4991  0.163795
##build_count_1946-1970                   4991  0.163795
##cafe_sum_1500_min_price_avg             4199  0.137803
##cafe_sum_1500_max_price_avg             4199  0.137803
##cafe_avg_price_1500                     4199  0.137803
##cafe_sum_2000_max_price_avg             1725  0.056611
##cafe_avg_price_2000                     1725  0.056611
##cafe_sum_2000_min_price_avg             1725  0.056611
##cafe_avg_price_3000                      991  0.032523
##cafe_sum_3000_max_price_avg              991  0.032523
##cafe_sum_3000_min_price_avg              991  0.032523
##cafe_avg_price_5000                      297  0.009747
##cafe_sum_5000_max_price_avg              297  0.009747
##cafe_sum_5000_min_price_avg              297  0.009747
##prom_part_5000                           178  0.005842
##floor                                    167  0.005481
##metro_min_walk                            25  0.000820
##railroad_station_walk_km                  25  0.000820
##railroad_station_walk_min                 25  0.000820
##ID_railroad_station_walk                  25  0.000820
##metro_km_walk                             25  0.000820
##*********************MISSING DATA TOP*****************



##all_data = all_data.fillna(all_data.mean())
##all_data['hospital_beds_raion'] = np.round(all_data['hospital_beds_raion'].fillna(all_data['hospital_beds_raion'].mean()))
##all_data = all_data.drop('build_year',1)
##all_data['state'] = np.round(all_data['state'].fillna(all_data['state'].mean()))
##all_data = all_data.drop('cafe_avg_price_500',1)
##all_data = all_data.drop('cafe_sum_500_max_price_avg',1)
##all_data = all_data.drop('cafe_sum_500_min_price_avg',1)
##all_data = all_data.drop('max_floor',1)
##all_data = all_data.drop('material',1)
##all_data['num_room'] = np.round(all_data['num_room'].fillna(all_data['num_room'].mean()))
##all_data['kitch_sq'] = np.round(all_data['kitch_sq'].fillna(all_data['kitch_sq'].mean()))
##all_data['preschool_quota'] = np.round(all_data['preschool_quota'].fillna(all_data['preschool_quota'].mean()))
##all_data['school_quota'] = np.round(all_data['school_quota'].fillna(all_data['school_quota'].mean()))
##all_data = all_data.drop('cafe_avg_price_1000',1)
##all_data = all_data.drop('cafe_sum_1000_max_price_avg',1)
##all_data = all_data.drop('cafe_sum_1000_min_price_avg',1)
##all_data['life_sq'] = np.round(all_data['life_sq'].fillna(all_data['life_sq'].mean()))
##all_data = all_data.drop('build_count_frame',1)
##all_data = all_data.drop('build_count_1971-1995',1)
##all_data = all_data.drop('build_count_block',1)
##all_data = all_data.drop('build_count_wood',1)
##all_data = all_data.drop('build_count_mix',1)
##all_data = all_data.drop('build_count_1921-1945',1)
##all_data = all_data.drop('build_count_panel',1)
##all_data = all_data.drop('build_count_foam',1)
##all_data = all_data.drop('build_count_slag',1)
##all_data = all_data.drop('raion_build_count_with_builddate_info',1)
##all_data = all_data.drop('build_count_monolith',1)
##all_data = all_data.drop('build_count_before_1920',1)
##all_data = all_data.drop('build_count_1946-1970',1)
##all_data = all_data.drop('build_count_brick',1)
##all_data = all_data.drop('raion_build_count_with_material_info',1)
##all_data = all_data.drop('build_count_after_1995',1)
##
##all_data = all_data.drop('cafe_sum_1500_min_price_avg',1)
##all_data = all_data.drop('cafe_sum_1500_max_price_avg',1)
##all_data = all_data.drop('cafe_avg_price_1500',1)
##all_data = all_data.drop('cafe_sum_2000_max_price_avg',1)
##all_data = all_data.drop('cafe_avg_price_2000',1)
##all_data = all_data.drop('cafe_sum_2000_min_price_avg',1)
##all_data = all_data.drop('cafe_avg_price_3000',1)
##all_data = all_data.drop('cafe_sum_3000_max_price_avg',1)
##all_data = all_data.drop('cafe_sum_3000_min_price_avg',1)
##all_data = all_data.drop('cafe_avg_price_5000',1)
##all_data = all_data.drop('cafe_sum_5000_max_price_avg',1)
##all_data = all_data.drop('cafe_sum_5000_min_price_avg',1)
##
##all_data['prom_part_5000'] = all_data['prom_part_5000'].fillna(all_data['prom_part_5000'].mean())
##all_data['floor'] = np.round(all_data['floor'].fillna(all_data['floor'].mean()))
##all_data['metro_min_walk'] = np.round(all_data['metro_min_walk'].fillna(all_data['metro_min_walk'].mean()))
##all_data['railroad_station_walk_km'] = np.round(all_data['railroad_station_walk_km'].fillna(all_data['railroad_station_walk_km'].mean()))
##all_data['railroad_station_walk_min'] = np.round(all_data['railroad_station_walk_min'].fillna(all_data['railroad_station_walk_min'].mean()))
##all_data = all_data.drop('ID_railroad_station_walk',1)
##all_data['metro_km_walk'] = np.round(all_data['metro_km_walk'].fillna(all_data['metro_km_walk'].mean()))
##
####DROP THE COLUMNS THAT DO NOT HAVE na BUT NOT RELEVANT TO OUR ANALYSIS
##all_data = all_data.drop('green_zone_part',1)
##all_data = all_data.drop('indust_part',1)
##
##
##all_data = all_data.drop('thermal_power_plant_raion',1)
##all_data = all_data.drop('incineration_raion',1)
##all_data = all_data.drop('oil_chemistry_raion',1)
##all_data = all_data.drop('radiation_raion',1)
##all_data = all_data.drop('railroad_terminal_raion',1)
##all_data = all_data.drop('big_market_raion',1)
##all_data = all_data.drop('nuclear_reactor_raion',1)
##all_data = all_data.drop('detention_facility_raion',1)
##
##
##
##all_data = all_data.drop('full_all',1)
##all_data = all_data.drop('male_f',1)
##all_data = all_data.drop('female_f',1)
##all_data = all_data.drop('young_all',1)
##all_data = all_data.drop('work_all',1)
##all_data = all_data.drop('ekder_all',1)
##all_data = all_data.drop('0_6_all',1)
##all_data = all_data.drop('7_14_all',1)
##all_data = all_data.drop('0_17_all',1)
##all_data = all_data.drop('0_17_male',1)
##all_data = all_data.drop('0_17_female',1)
##all_data = all_data.drop('16_29_all',1)
##all_data = all_data.drop('0_13_all',1)
##all_data = all_data.drop('0_13_male',1)
##all_data = all_data.drop('0_13_female',1)
##
##all_data = all_data.drop('metro_min_avto',1)
##all_data = all_data.drop('metro_min_walk',1)
##all_data = all_data.drop('railroad_station_walk_min',1)
##all_data = all_data.drop('railroad_station_avto_min',1)
##all_data = all_data.drop('ID_railroad_station_avto',1)
##all_data = all_data.drop('public_transport_station_min_walk',1)
##all_data = all_data.drop('water_1line',1)
##all_data = all_data.drop('ID_big_road1',1)
##all_data = all_data.drop('ID_big_road2',1)
##all_data = all_data.drop('ID_railroad_terminal',1)
##all_data = all_data.drop('ID_bus_terminal',1)
##
##all_data = all_data.drop('ice_rink_km',1)
##all_data = all_data.drop('basketball_km',1)
##all_data = all_data.drop('swim_pool_km',1)
##all_data = all_data.drop('detention_facility_km',1)
##all_data = all_data.drop('ecology',1)
##
##for col in all_data.columns:
##    if 'cafe_couunt_500_' in col:
##        del all_data[col]
##
##for col in all_data.columns:
##    if 'cafe_couunt_1000_' in col:
##        del all_data[col]
##
##for col in all_data.columns:
##    if 'cafe_couunt_1500_' in col:
##        del all_data[col]
##
##for col in all_data.columns:
##    if 'cafe_couunt_2000_' in col:
##        del all_data[col]        
##
##for col in all_data.columns:
##    if 'cafe_couunt_3000_' in col:
##        del all_data[col]        
##
##for col in all_data.columns:
##    if 'cafe_couunt_5000_' in col:
##        del all_data[col]


##*****************all_data missing data***********************
##                                       Total   Percent
##hospital_beds_raion                    17859  0.468335 
##cafe_sum_500_min_price_avg             16440  0.431123
##cafe_sum_500_max_price_avg             16440  0.431123
##cafe_avg_price_500                     16440  0.431123
##build_year                             14654  0.384287
##state                                  14253  0.373771
##max_floor                               9572  0.251016
##material                                9572  0.251016
##num_room                                9572  0.251016
##kitch_sq                                9572  0.251016
##preschool_quota                         8284  0.217240
##school_quota                            8280  0.217135
##cafe_avg_price_1000                     7746  0.203131
##cafe_sum_1000_max_price_avg             7746  0.203131
##cafe_sum_1000_min_price_avg             7746  0.203131
##life_sq                                 7559  0.198227
##build_count_frame                       6209  0.162825
##build_count_foam                        6209  0.162825
##build_count_brick                       6209  0.162825
##build_count_monolith                    6209  0.162825
##build_count_panel                       6209  0.162825
##raion_build_count_with_material_info    6209  0.162825
##build_count_after_1995                  6209  0.162825
##build_count_1921-1945                   6209  0.162825
##build_count_slag                        6209  0.162825
##build_count_mix                         6209  0.162825
##raion_build_count_with_builddate_info   6209  0.162825
##build_count_before_1920                 6209  0.162825
##build_count_wood                        6209  0.162825
##build_count_1946-1970                   6209  0.162825
##build_count_1971-1995                   6209  0.162825
##build_count_block                       6209  0.162825
##cafe_sum_1500_max_price_avg             5020  0.131645
##cafe_avg_price_1500                     5020  0.131645
##cafe_sum_1500_min_price_avg             5020  0.131645
##cafe_avg_price_2000                     2149  0.056355
##cafe_sum_2000_max_price_avg             2149  0.056355
##cafe_sum_2000_min_price_avg             2149  0.056355
##cafe_sum_3000_min_price_avg             1173  0.030761
##cafe_avg_price_3000                     1173  0.030761
##cafe_sum_3000_max_price_avg             1173  0.030761
##cafe_sum_5000_min_price_avg              425  0.011145
##cafe_sum_5000_max_price_avg              425  0.011145
##cafe_avg_price_5000                      425  0.011145
##prom_part_5000                           270  0.007080
##floor                                    167  0.004379
##metro_min_walk                            59  0.001547
##railroad_station_walk_min                 59  0.001547
##metro_km_walk                             59  0.001547
##ID_railroad_station_walk                  59  0.001547
##railroad_station_walk_km                  59  0.001547
##product_type                              33  0.000865
##green_part_2000                           19  0.000498

##*****************all_data missing data***********************

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

print(all_data.isnull().sum().max()) #just checking that there's no missing data missing...

##X_train = StandardScaler().fit_transform(X_train)
##y = StandardScaler().fit_transform(np.reshape(df_train['price_doc'].values,(-1,1)));
####print(X_train)
##low_range = y[y[:,0].argsort()][:10]
##high_range= y[y[:,0].argsort()][-10:]
##print('outer range (low) of the distribution:')
##print(low_range)
##print('\nouter range (high) of the distribution:')
##print(high_range)

###bivariate analysis saleprice/grlivarea
##var = 'full_sq'
##data = pd.concat([df_train['price_doc'], df_train[var]], axis=1)
##data.plot.scatter(x=var, y='price_doc', ylim=(0,10000000), xlim=(0,200));
##plt.show()

#histogram and normal probability plot
##sns.distplot(df_train['price_doc'], fit=norm);
##fig = plt.figure()
##res = stats.probplot(df_train['price_doc'], plot=plt)
##plt.show()


##applying log transformation: It shows 'peakedness', positive skewness and does not follow the diagonal line.
##in case of positive skewness, log transformations usually works well.
df_train['price_doc'] = np.log1p(df_train['price_doc'])

##transformed histogram and normal probability plot
##sns.distplot(df_train['price_doc'], fit=norm);
##fig = plt.figure()
##res = stats.probplot(df_train['price_doc'], plot=plt)
##plt.show()
numeric_features = all_data.dtypes[all_data.dtypes != "object"].index
skewed_features = df_train[numeric_features].apply(lambda x: skew(x.dropna())) #compute skewness
##*******************TODO: CHECK IF ABOVE USE "all_data[numeric..." OR "train_df[numeric....."***********************
skewed_features = skewed_features.index
all_data[skewed_features] = np.log1p(all_data[skewed_features])
##all_data = pd.get_dummies(all_data)
all_data = pd.get_dummies(all_data)


X_train = all_data[:df_train.shape[0]]
y = df_train.price_doc
##print(X_train.shape)#(30471, 1779)
##print(y.shape)#(30471,)
X_test = all_data[df_train.shape[0]:]
##print(X_test.shape)#(7662, 1779)
dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)


##eta: 0.01 - 0.3
##max_depth: 3 - 10
##gamma: 0 - 0.2
##subsample: 0.6 - 0.9
##colsample_bytree: 0.5 - 0.8
##alpha: 1

LowestError = 10000


##eta = {}




params = {"max_depth":3, "eta":1, "alpha": 1, "colsample_bytree":0.5, "subsample": 0.5 }
bestParams = params
bestBoostRound = 100
##for a in np.arange(0.01,0.3,0.01):
    
for currentDep in np.arange(3, 9, 1):
    for currentEta in np.arange(0.01, 0.31, 0.1):
        for currentColsampleByTree in np.arange(0.3,0.9,0.1):
            for currentSubsample in np.arange(0.3,0.9,0.1):
                for currentGamma in np.arange(0,0.21,0.1):
                    for currentBoostRound in np.arange(100, 1010, 10):
                        currentParams = {"max_depth":currentDep, "gamma": currentGamma ,"eta":currentEta, "alpha": 1, "colsample_bytree":currentColsampleByTree, "subsample": currentSubsample }
                        print("NEW CROSS VALIDATION RUNNING....")

                        model = xgb.cv(currentParams, dtrain,  num_boost_round=currentBoostRound, early_stopping_rounds=8, nfold = 5)
                        

                        if model.values[model.shape[0]-1][0] < LowestError:
                            LowestError = model.values[model.shape[0]-1][0]
                            bestBoostRound = currentBoostRound
                            bestParams = currentParams
                        print("LowestError: ", LowestError)
                        print("Best BoostRound: ", bestBoostRound)
                        print("Best Parameter: ", bestParams )



## random.uniform(1.5, 1.9)
##model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

##print(model.values)##[173 rows x 4 columns]
##print(type(model.values))
##print(model.values.shape)
##print(model.values[model.shape[0]-1][0])

##print(model.shape)(173, 4)
##print(type(model))<class 'pandas.core.frame.DataFrame'>


##model_xgb = xgb.XGBRegressor(
##    n_estimators=1000,
##    max_depth=5,
##    learning_rate=0.2,reg_alpha=1,subsample=0.75, colsample_bytree=0.75 ) #the params were tuned using xgb.cv
##
##
##model_xgb.fit(X_train, y)
####xgb_preds = np.expm1(model_xgb.predict(X_test))
##
####xgb_preds = model_xgb.predict(X_test)
##
##xgb_preds = np.expm1(model_xgb.predict(X_test))
##preds = xgb_preds
##solution = pd.DataFrame({"id":df_test.id, "price_doc":preds})
##solution.to_csv("RussianHousingPredict.csv", index = False)



##saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);










































        
##print(all_data['ecology'].loc[all_data['ecology'] == 'no data'])##7656 OUR OF 30471 = 25% MISSING

##print(all_data['kitch_sq'])
##print(temp)
##print(all_data['state'])



##all_data = all_data.drop((missing_data[missing_data['Total'] > 1]).index,1)
##print(all_data.isnull().sum().max()) #just checking that there's no missing data missing...



##print(all_data.shape)##(30471, 290)
##print(all_data.shape)##(30471, 290)

##print(type(all_data))##<class 'pandas.core.frame.DataFrame'>



print("END OF PROGRAM")
