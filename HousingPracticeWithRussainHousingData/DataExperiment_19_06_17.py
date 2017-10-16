import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
##from sklearn import model_selection, preprocessing
import xgboost as xgb


color = sns.color_palette()

macro_cols = ["balance_trade",
              "balance_trade_growth", "eurrub",
              "average_provision_of_build_contract",
              "micex_rgbi_tr", "micex_cbi_tr",
              "deposits_rate", "mortgage_value",
              "mortgage_rate",
              "income_per_cap", "rent_price_4+room_bus",
              "museum_visitis_per_100_cap", "apartment_build"]

pd.options.mode.chained_assignment = None  # default='warn'
##pd.set_option('display.max_columns', 500)

##train_df = pd.read_csv("train_clean.csv", parse_dates=['timestamp'])

train_df = pd.read_csv("train.csv", parse_dates=['timestamp'])

##test_df = pd.read_csv("test_clean.csv", parse_dates=['timestamp'])

test_df = pd.read_csv("test.csv", parse_dates=['timestamp'])
macro_df = pd.read_csv("macro.csv", parse_dates=['timestamp'],
                       usecols=['timestamp'] + macro_cols)

macro_df_all = pd.read_csv("macro.csv", parse_dates=['timestamp'])
                           


all_data = pd.concat((train_df.loc[:,'timestamp':'market_count_5000'],
                      test_df.loc[:,'timestamp':'market_count_5000']))

##test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')
##train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')

##%%%%%%%%%%%%%%%%%%%%%GET ROWS THAT HAVE NAN BUILD_YEARS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_df_build_year = train_df.build_year
##print(train_df_build_year)
train_df_build_year_NAN_index = train_df['build_year'].index[train_df['build_year'].apply(np.isnan)]
##train_df_build_year_nonNAN_rows = train_df['build_year'][train_df_build_year_notNAN_index]
train_df_build_year_NAN_rows = train_df.drop('build_year', 1).iloc[train_df_build_year_NAN_index]

train_df_build_year_NAN_rows_withBuildYear = train_df.iloc[train_df_build_year_NAN_index]

##print(train_df_build_year_NAN_index)
##print(type(train_df_build_year_NAN_index))##<class 'pandas.core.indexes.numeric.Int64Index'>
##print(train_df_build_year_NAN_rows)
##print(train_df_build_year_NAN_rows.describe())
##print(len(list(train_df_build_year_NAN_rows.columns.values)))##291


##train_df_build_year_NAN_rows_floorNAN_Index = train_df_build_year_NAN_rows['floor'].index[train_df_build_year_NAN_rows['floor'].apply(np.isnan)]
##train_df_build_year_NAN_rows_floorNAN_rows = train_df_build_year_NAN_rows.iloc[train_df_build_year_NAN_rows_floorNAN_Index]
##print(train_df_build_year_NAN_rows_floorNAN_Index)
##print(train_df_build_year_NAN_rows_floorNAN_rows.values.)



##print(train_df_build_year_NAN_rows.values.shape)##(14505, 291)
##train_df_build_year_NAN_rows_without_NAN = train_df_build_year_NAN_rows.dropna()
##print(train_df_build_year_NAN_rows_without_NAN.values.shape)##(45, 291)






##%%%%%%%%%%%%%%%%%%%%%GET ROWS THAT HAVE NO NAN%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

##train_df_without_NAN = train_df[train_df.notnull()]##<----THIS ONE DOESN'T WORK
train_df_without_NAN = train_df.dropna()
##train_df_without_NAN = train_df_without_NAN.drop(["id", "timestamp", "price_doc"], 1)
test_df_without_NAN = test_df.dropna()
macro_df_without_NAN = macro_df.dropna()
macro_df_all_without_NAN = macro_df_all.dropna()
print(macro_df_all_without_NAN.describe())

##test_df_without_NAN = test_df_without_NAN.drop(["id", "timestamp"], 1)
##print(train_df_without_NAN2)
##print(train_df_without_NAN2.values.shape)##(5662, 292)####<------ONLY 5662 out of 30000+ all complete

all_data_without_NAN = pd.concat((train_df_without_NAN.loc[:,'timestamp':'market_count_5000'],
                      test_df_without_NAN.loc[:,'timestamp':'market_count_5000']))

##print(all_data_without_NAN['state'].describe())
##print(all_data_without_NAN['school_quota'].describe())


test_df_without_NAN_macro = pd.merge(test_df_without_NAN, macro_df_all_without_NAN, how='left', on='timestamp')

train_df_without_NAN_macro = pd.merge(train_df_without_NAN, macro_df_all_without_NAN, how='left', on='timestamp')


all_data_without_NAN_macro = pd.concat((train_df_without_NAN_macro.loc[:,'timestamp':'hospital_bed_occupancy_per_year'],
                      train_df_without_NAN_macro.loc[:,'timestamp':'hospital_bed_occupancy_per_year']))




##                      test_df_without_NAN_macro.loc[:,'timestamp':'apartment_fund_sqm']))
##train_df_without_NAN_columns = train_df.drop(["id", "timestamp", "price_doc"], 1).dropna(axis=1)
##print(train_df_without_NAN_columns.isnull().values.any())##False

##test_df_without_NAN_columns = test_df.drop(["id", "timestamp"] , axis=1).dropna(axis=1)
##print(test_df_without_NAN_columns.isnull().values.any())##False


##print(train_df_without_NAN_columns.values.shape)##(30471, 239)
##print(test_df_without_NAN_columns.values.shape)##(7662, 239)




##train_df_without_NAN_columns_names = train_df_without_NAN_columns.columns.values
##print(train_df_without_NAN_columns_names.shape)
##test_df_without_NAN_columns_names = test_df_without_NAN_columns.columns.values
##print(test_df_without_NAN_columns_names.shape)

##print(np.sum(train_df_without_NAN_columns_names == test_df_without_NAN_columns_names))
##print(train_df_without_NAN_columns_names[train_df_without_NAN_columns_names != test_df_without_NAN_columns_names])
##print(train_df_without_NAN_columns_names)
##print(test_df_without_NAN_columns_names)
##print(np.array_equal(train_df_without_NAN_columns_names , test_df_without_NAN_columns_names))

##print(type(train_df_without_NAN_columns_names))##<class 'numpy.ndarray'>
##print(np.subtract(train_df_without_NAN_columns_names, test_df_without_NAN_columns_names))



##print(all_data_without_NAN.values.shape)##(7859, 290)
##print(all_data_without_NAN.isnull().values.any())##False









##%%%%%%%%%%%%%%%%%%%%%FIND CORRELATION BETWEEN BUILD_YEAR AND EVERYTHING ELSE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


####correlation matrix


####scatter plot price/build_year
##var = 'hospital_beds_available_per_cap'##life_sq
##var = 'hospital_beds_raion'
##data = pd.concat([all_data_without_NAN_macro['hospital_beds_available_per_cap'], all_data_without_NAN_macro[var]], axis=1)
##data.plot.scatter(x=var, y='hospital_beds_available_per_cap', xlim=(0,800), ylim=(20,5000));
##plt.show()





##all_data_corrmat = all_data_without_NAN.corr()
####train_data_corrmat = train_df_without_NAN.corr()
##k = 10 #number of variables for heatmap
##cols = all_data_corrmat.nlargest(k, 'build_count_frame')['build_count_frame'].index
####cols = train_data_corrmat.nlargest(k, 'price_doc')['price_doc'].index
####cols = corrmat.nlargest(k, 'kitch_sq')['kitch_sq'].index
##cm = np.corrcoef(all_data_corrmat[cols].values.T)
####cm = np.corrcoef(train_data_corrmat[cols].values.T)
##sns.set(font_scale=1.25)
##hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
##plt.yticks(rotation=0)
##plt.xticks(rotation=90)
##plt.show()
##sns.plt.show()

##all_data_macro_corrmat = macro_df_all_without_NAN.corr()
####train_data_corrmat = train_df_without_NAN.corr()
##k = 10 #number of variables for heatmap
##cols = all_data_macro_corrmat.nlargest(k, 'salary')['salary'].index
####cols = train_data_corrmat.nlargest(k, 'price_doc')['price_doc'].index
####cols = corrmat.nlargest(k, 'kitch_sq')['kitch_sq'].index
##cm = np.corrcoef(all_data_macro_corrmat[cols].values.T)
####cm = np.corrcoef(train_data_corrmat[cols].values.T)
##sns.set(font_scale=1.25)
##hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
##plt.yticks(rotation=0)
##plt.xticks(rotation=90)
##plt.show()
##sns.plt.show()


##macro_data_corrmat = macro_df_all_without_NAN.corr()
##k = 10 #number of variables for heatmap
##cols = macro_data_corrmat.nlargest(k, 'salary')['salary'].index
##cm = np.corrcoef(macro_data_corrmat[cols].values.T)
##sns.set(font_scale=1.25)
##hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
##plt.yticks(rotation=0)
##plt.xticks(rotation=90)
##plt.show()
##sns.plt.show()

##MOST CORRELATED COLUMNS:
##max_floor: 0.84, floor: 0.75, bulvar_ring_km: 0.79, zd_vokzaly_avto_km: 0.78, sadovoe_km: 0.79, kremlin_km: 0.79, ttk_km: 0.77, 

DataMostCorrelatedToBuildYear = all_data_without_NAN[['max_floor', 'floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km',
                                                     'kremlin_km', 'ttk_km']]##<------- only need to fill max_floor and floor

DataMostCorrelatedTo_max_floor = all_data_without_NAN[['build_year', 'floor', 'kitch_sq']]##<------- only need to fill max_floor and floor
##DataMostCorrelatedTo_Max_Floor
##print(DataMostCorrelatedToBuildYear.values.shape)##(7859, 7)




##%%%%%%%%%%%%%%%%%%%%%FIX DATA, OR ADD MISSING DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#######FIXING build_year
#######FIXING build_year
#######FIXING build_year
#######FIXING build_year
#######FIXING build_year
##df["column"].fillna(lambda x: random.choice(df[df[column] != np.nan]["column"]), inplace =True)
##print(all_data['floor'].unique())
##df.loc[pd.isnull(df.index)]
all_data_build_year_NAN_index = all_data.index[all_data['build_year'].apply(np.isnan)]


all_data_build_year_NAN_rows = all_data['build_year'].loc[all_data['build_year'].index.isin(all_data_build_year_NAN_index)]

##all_data_build_year_NAN_rows2 = all_data['build_year'].iloc[all_data['build_year'].index.isin(all_data_build_year_NAN_index)]
##all_data_build_year_NAN_index2 = all_data.index[all_data['build_year'] == np.nan]<---------------THIS ONE DOESN'T WORK!

##print(all_data['floor'].loc[~all_data['floor'].index.isin(all_data_build_year_NAN_index)])##Length: 37830
##print(all_data['floor'].loc[~all_data['floor'].index.isin(all_data_build_year_NAN_index)].unique())
##df.loc[~df.index.isin(t)]
##df[df[column] != np.nan]["column"]
##print(all_data.values.shape)##(38133, 290)
##print(all_data['build_year'].loc[pd.isnull(all_data['build_year'].index)])
##print(pd.isnull(all_data['build_year'].index).shape)##(38133,)
##print(all_data_build_year_NAN_index)##length=177
##print(all_data['build_year'].unique())
##print(all_data_build_year_NAN_rows.unique())
##print(all_data_build_year_NAN_rows)
##all_data_build_year_NOTNAN_rows = all_data.iloc[all_data_build_year_NOTNAN_index]['floor']
##print(all_data_build_year_NOTNAN_rows)
all_data['floor'] = all_data['floor'].fillna(all_data['floor'].median())
##print(all_data['floor'].unique())

##print(all_data.iloc[all_data_build_year_NAN_index]['floor'])

##test_X = all_data[['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km',
##                                                     'kremlin_km', 'ttk_km']].values
##
##test_X = all_data[['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km',
##                                                     'kremlin_km', 'ttk_km']].loc[~all_data.index.isin(all_data_build_year_NAN_index)].values
##
##
##test_build_year = all_data.loc[~all_data.index.isin(all_data_build_year_NAN_index)]
##print(test_X.shape)##(15966, 6)
##print(test_build_year.values.shape)



##
##train_df_build_year_NAN_index = train_df['build_year'].index[train_df['build_year'].apply(np.isnan)]
##train_df_build_year_NOTNAN_index = train_df['build_year'].index[train_df['build_year'].apply(np.isfinite)]
##all_data_build_year_NOTNAN_index = train_df['build_year'].index[train_df['build_year'].apply(np.isfinite)]
####print(train_df_build_year_NAN_index)
##print(train_df_build_year_NOTNAN_index)
##train_df_build_year_NAN_rows = train_df.drop('build_year', 1).iloc[train_df_build_year_NAN_index]
##train_df_build_year_NAN_rows  = train_df_build_year_NAN_rows[['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km','kremlin_km', 'ttk_km']]
##print(train_df_build_year_NAN_rows.values.shape)##(14505, 6)
##test_X = train_df_build_year_NAN_rows.values                                          
##
##
##
####print(all_data['floor'])
####print(all_data['floor'].isnull().values.any())##False
####print(all_data['floor'].describe())
##
##train_X = DataMostCorrelatedToBuildYear[['floor', 'bulvar_ring_km', 'zd_vokzaly_avto_km', 'sadovoe_km',
##                                                     'kremlin_km', 'ttk_km']].values
##print(train_X.shape)
##train_y = all_data_without_NAN['build_year']
##print(train_y.shape)
##
##dtrain = xgb.DMatrix(train_X, label = train_y)
##
##earlyStopRounds = 10
##currentParams = {"max_depth":5, "gamma": 0.2,
##                 "eta":0.01, "colsample_bytree":0.75,
##                 "subsample": 0.75, 'objective': 'reg:linear',
##                 'min_child_weight':1}
##SEED = 0
####print(currentParams)
##
####model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
####        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
####print("ERROR: ", model.values[model.shape[0]-1][0])
####print("BEST ROUND: ", model.shape[0])
##
##
##best_num_boost_round = 4363
##best_num_boost_round = int((best_num_boost_round - earlyStopRounds) / (1 - 1 / 5))
##model_xgb = xgb.XGBRegressor(
##    objective="reg:linear",
##    n_estimators=best_num_boost_round,
##    max_depth=5,
##    learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=1, seed=SEED,
##    gamma=0.2)
##
##model_xgb.fit(train_X, train_y)
##
##
##y_pred = model_xgb.predict(test_X)
##y_pred = np.around(y_pred)
##print(y_pred)
##print(y_pred.shape)
##print(train_df['build_year'][train_df_build_year_NAN_index]) 
##train_df['build_year'][train_df_build_year_NAN_index] = y_pred
##train_df['build_year'][train_df_build_year_NAN_index] += 5
##print(train_df['build_year'][train_df_build_year_NAN_index]) 

#######FIXING max_floor
#######FIXING max_floor
#######FIXING max_floor
#######FIXING max_floor
#######FIXING max_floor
























##%%%%%%%%%%%%%%%%%%%%%FIND THE MODEL THAT CAN PREDICT build_year with the MOST CORRELATED VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

##train_X = DataMostCorrelatedToBuildYear.values
##train_y = all_data_without_NAN['build_year']

##print(train_X.shape)##(7859, 7)
##train_y = all_data_without_NAN[]
##dtrain = xgb.DMatrix(train_X, label = train_y)

##earlyStopRounds = 10
##currentParams = {"max_depth":4, "gamma": 0.2,
##                 "eta":0.02, "colsample_bytree":0.8,
##                 "subsample": 0.7, 'objective': 'reg:linear',
##                 'min_child_weight':1}
##SEED = 0
##
##print(currentParams)
##model = xgb.cv(currentParams, dtrain,  num_boost_round=10000000000,
##        early_stopping_rounds= earlyStopRounds, nfold = 5, verbose_eval=5, seed=SEED)
##print("ERROR: ", model.values[model.shape[0]-1][0])
##print("BEST ROUND: ", model.shape[0])













print("END OF PROGRAM")

