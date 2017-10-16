import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn import model_selection, preprocessing
import xgboost as xgb

import tensorflow as tf
from tensorflow.python.client import device_lib

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

color = sns.color_palette()

# From here: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity/notebook
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)

train_df = pd.read_csv("train_clean.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("test_clean.csv", parse_dates=['timestamp'])
##macro_df = pd.read_csv("macro.csv", parse_dates=['timestamp'])
macro_df = pd.read_csv("macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

##print(macro_df.describe())
test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')
train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')
##print(train_df.shape, test_df.shape)

id_test = test_df['id']

# truncate the extreme values in price_doc #
ulimit = np.percentile(train_df.price_doc.values, 99)
llimit = np.percentile(train_df.price_doc.values, 1)
train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit

##print(train_df.shape)
##print(test_df.shape)
#['open', 'close']
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
                        'full_all', 'young_all','work_all',
                       'ekder_all','0_6_all', '7_14_all','0_17_all','16_29_all',
                       '0_13_all'
                       ,'eurrub','ekder_all'
                
                    
                        
                        ]
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



train_df.sub_area = train_df.sub_area.fillna(train_df.sub_area.mode())
##print(train_df.sub_area)
test_df.sub_area = test_df.sub_area.fillna(train_df.sub_area.mode())
print(np.unique(train_df.sub_area).shape)
print(np.unique(test_df.sub_area).shape)
print(np.setdiff1d(np.unique(train_df.sub_area), np.unique(test_df.sub_area)))
##print(train_df.sub_area.describe())
##print(test_df.sub_area.describe())


train_df = train_df.drop(listOfDroppedColumns3,1)
test_df = test_df.drop(listOfDroppedColumns3,1)

train_df.ecology[train_df.ecology == 'excellent'] = 4
train_df.ecology[train_df.ecology == 'good'] = 3
train_df.ecology[train_df.ecology == 'satisfactory'] = 2
train_df.ecology[train_df.ecology == 'poor'] = 1
train_df.ecology[train_df.ecology == 'no data'] = 0
##pd.to_numeric(train_df.ecology)
train_df.ecology = train_df.ecology.astype('int')
train_df.ecology[train_df.ecology == 0] = train_df.ecology.median()


test_df.ecology[test_df.ecology == 'excellent'] = 4
test_df.ecology[test_df.ecology == 'good'] = 3
test_df.ecology[test_df.ecology == 'satisfactory'] = 2
test_df.ecology[test_df.ecology == 'poor'] = 1
test_df.ecology[test_df.ecology == 'no data'] = 0
##pd.to_numeric(test_df.ecology)
test_df.ecology = train_df.ecology.astype('int')
test_df.ecology[test_df.ecology == 0] = test_df.ecology.median()








train_df_withoutTS = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
test_df_withoutTS = test_df.drop(["id", "timestamp"] , axis=1)


train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)
train_df = train_df.fillna(train_df.median())
##train_df = train_df.fillna(train_df.mode())

test_df = test_df.fillna(test_df.median())

##test_df = test_df.fillna(test_df.mode())


print(np.setdiff1d(train_df.columns.values, test_df.columns.values))##'sub_area_Poselenie Klenovskoe'
print(np.where(train_df.columns.values == 'sub_area_Poselenie Klenovskoe'))##(array([368], dtype=int64),)
##sub_area_Poselenie_Klenovskoe_column=pd.Series(np.zeros(7662))
train_df = train_df.drop('sub_area_Poselenie Klenovskoe', 1)
print(np.setdiff1d(train_df.columns.values, test_df.columns.values))##'sub_area_Poselenie Klenovskoe'




# year and month #
train_df["yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month
test_df["yearmonth"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.month
##
# year and week #
train_df["yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear
test_df["yearweek"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.weekofyear
##
# year #
train_df["year"] = train_df["timestamp"].dt.year
test_df["year"] = test_df["timestamp"].dt.year


train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
train_X = train_X.values
test_X = test_df.drop(["id", "timestamp"] , axis=1)
test_X = test_X.values
train_y = np.log1p(train_df.price_doc.values)


print(train_X.shape)##(30471, 243)
##print(type(train_X))##<class 'pandas.core.frame.DataFrame'>
print(test_X.shape)##(7662, 243)
##print(type(test_X))##<class 'pandas.core.frame.DataFrame'>
print(train_y.shape)##(30471,)
##print(type(train_y))


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

##BELOW IS THE TENSOR FLOW PART OF THIS PROGRAM
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 448])##holder for training data
y_ = tf.placeholder(tf.float32, shape=[None, 1])##holder for truth values
keep_prob_fc1 = tf.placeholder(tf.float32)

W_fc1 = weight_variable([448,448])##243
b_fc1 = bias_variable([448])
fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = weight_variable([448, 448 * 2])
b_fc2 = bias_variable([448 * 2])
fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)
##fc2 = tf.nn.dropout(fc2, keep_prob_fc1)

W_fc3 = weight_variable([448 * 2, 448 * 2])
b_fc3 = bias_variable([448 * 2])
fc3 = tf.nn.relu(tf.matmul(fc2, W_fc3) + b_fc3)

W_fc4 = weight_variable([448 * 2, 448 * 2])
b_fc4 = bias_variable([448 * 2])
fc4 = tf.nn.relu(tf.matmul(fc3, W_fc4) + b_fc4)

W_fc5 = weight_variable([448 * 2, 1])
b_fc5 = bias_variable([1])

y_output = tf.nn.relu(tf.matmul(fc4, W_fc5) + b_fc5)

rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, y_output))))

train_step = tf.train.AdamOptimizer(1e-4).minimize(rmse)

sess.run(tf.global_variables_initializer())

for i in range(200000):##25000

    Xtr, Ytr = next_batch(1000, train_X, train_y)
 
    Ytr = np.reshape(Ytr,(1000,1))
    if i%100 == 0:
      train_error = sess.run(rmse, feed_dict={x:Xtr, y_:Ytr, keep_prob_fc1: 1})
      print("step %d, training error %g"%(i, train_error))
    train_step.run(feed_dict={x: Xtr, y_: Ytr, keep_prob_fc1: 0.8})



print("END OF PROGRAM")
