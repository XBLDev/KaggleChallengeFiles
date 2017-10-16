import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
train = pd.read_csv("train.csv", encoding= "utf_8")
test = pd.read_csv("test.csv", encoding= "utf_8")

first_feat = ["id","timestamp","price_doc", "full_sq", "life_sq",
"floor", "max_floor", "material", "build_year", "num_room",
"kitch_sq", "state", "product_type", "sub_area"]

##BELOW DATA CLEAN FROM: https://www.kaggle.com/nigelcarpenter/cleaning-the-data-using-latitude-and-longitude

bad_index = train[train.build_year == 1691].index
##print(train.ix[bad_index, ["build_year"]])
train.ix[bad_index, "build_year"] = 1961
bad_index = test[test.build_year == 1691].index
##print(test.ix[bad_index, ["build_year"]])
test.ix[bad_index, "build_year"] = 1961

bad_index = train[train.build_year == 215].index
##print(train.ix[bad_index, ["build_year"]])
train.ix[bad_index, "build_year"] = 2015
bad_index = test[test.build_year == 215].index
##print(test.ix[bad_index, ["build_year"]])
test.ix[bad_index, "build_year"] = 2015

bad_index = train[train.build_year == 4965].index
##print(train.ix[bad_index, ["build_year"]])
train.ix[bad_index, "build_year"] = 1965
bad_index = test[test.build_year == 4965].index
##print(test.ix[bad_index, ["build_year"]])
test.ix[bad_index, "build_year"] = 1965
##
bad_index = train[train.build_year == 2].index
##print(train.ix[bad_index, ["build_year"]])
train.ix[bad_index, "build_year"] = 2014
bad_index = test[test.build_year == 2].index
##print(test.ix[bad_index, ["build_year"]])
test.ix[bad_index, "build_year"] = 2014

bad_index = train[train.build_year == 3].index
train.ix[bad_index, "build_year"][0] = 2013
train.ix[bad_index, "build_year"][1] = 2013

bad_index = train[train.build_year == 20].index
##print(train.ix[bad_index, ["build_year"]])
train.ix[bad_index, "build_year"] = 2014
bad_index = test[test.build_year == 20].index
##print(test.ix[bad_index, ["build_year"]])
test.ix[bad_index, "build_year"] = 2014

bad_index = train[train.build_year == 20052009].index
##print(train.ix[bad_index, ["build_year"]])
train.ix[bad_index, "build_year"] = 2009

bad_index = train[train.build_year == 0].index
##print(train.ix[bad_index, ["build_year"]])
train.ix[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year == 0].index
##print(train.ix[bad_index, ["build_year"]])
test.ix[bad_index, "build_year"] = np.NaN

bad_index = train[train.build_year == 1].index
##print(train.ix[bad_index, ["build_year"]])
train.ix[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year == 1].index
##print(train.ix[bad_index, ["build_year"]])
test.ix[bad_index, "build_year"] = np.NaN

bad_index = train[train.build_year == 71].index
##print(train.ix[bad_index, ["build_year"]])
train.ix[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year == 71].index
##print(train.ix[bad_index, ["build_year"]])
test.ix[bad_index, "build_year"] = np.NaN

##BELOW DATA CLEAN FROM: https://www.kaggle.com/keremt/very-extensive-cleaning-by-sberbank-discussions/notebook

bad_index = train[train.life_sq > train.full_sq].index
train.ix[bad_index, "life_sq"] = np.NaN

equal_index = [601,1896,2791]
test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]

bad_index = test[test.life_sq > test.full_sq].index
test.ix[bad_index, "life_sq"] = np.NaN

bad_index = train[train.life_sq < 5].index
train.ix[bad_index, "life_sq"] = np.NaN

bad_index = test[test.life_sq < 5].index
test.ix[bad_index, "life_sq"] = np.NaN

bad_index = train[train.full_sq < 10].index
train.ix[bad_index, "full_sq"] = np.NaN

bad_index = test[test.full_sq < 10].index
test.ix[bad_index, "full_sq"] = np.NaN

bad_index = train[(train.full_sq > 250) & (train.full_sq <= 1000)].index
##print(train.ix[bad_index, "full_sq"])
train.ix[bad_index, "full_sq"] = train.ix[bad_index, "full_sq"]/10

bad_index = test[(test.full_sq > 250) & (test.full_sq <= 1000)].index
##print(test.ix[bad_index, "full_sq"])
test.ix[bad_index, "full_sq"] = test.ix[bad_index, "full_sq"]/10

bad_index = train[train.full_sq > 1000].index
##print(train.ix[bad_index, "full_sq"])
train.ix[bad_index, "full_sq"] = train.ix[bad_index, "full_sq"]/100

bad_index = test[test.full_sq > 1000].index
##print(test.ix[bad_index, "full_sq"])
test.ix[bad_index, "full_sq"] = test.ix[bad_index, "full_sq"]/100


kitch_is_build_year = [13117]
##print(train.ix[kitch_is_build_year, "build_year"])
##print(train.ix[kitch_is_build_year, "kitch_sq"])
train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]



bad_index = train[train.kitch_sq >= train.life_sq].index
train.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = test[test.kitch_sq >= test.life_sq].index
test.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = train[(train.full_sq > 180) & (train.life_sq / train.full_sq < 0.3)].index
##print(train.ix[bad_index, "full_sq"])
train.ix[bad_index, "full_sq"] = np.NaN

bad_index = test[(test.full_sq > 180) & (test.life_sq / test.full_sq < 0.3)].index
test.ix[bad_index, "full_sq"] = np.NaN

bad_index = train[train.life_sq > 250].index
train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN

bad_index = test[test.life_sq > 250].index
test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN

bad_index = train[train.build_year < 1500].index
train.ix[bad_index, "build_year"] = np.NaN

bad_index = test[test.build_year < 1500].index
test.ix[bad_index, "build_year"] = np.NaN

bad_index = train[train.num_room == 0].index 
train.ix[bad_index, "num_room"] = np.NaN

bad_index = test[test.num_room == 0].index 
test.ix[bad_index, "num_room"] = np.NaN

bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
##print(train.ix[bad_index, "num_room"])
train.ix[bad_index, "num_room"] = np.NaN

bad_index = [3174, 7313]
##print(test.ix[bad_index, "num_room"])
test.ix[bad_index, "num_room"] = np.NaN

bad_index = train[(train.floor == 0).values & (train.max_floor == 0).values].index
##print(train.ix[bad_index, ["max_floor", "floor"]])
train.ix[bad_index, ["max_floor", "floor"]] = np.NaN

bad_index = train[train.floor == 0].index
train.ix[bad_index, "floor"] = np.NaN

bad_index = train[train.max_floor == 0].index
train.ix[bad_index, "max_floor"] = np.NaN

bad_index = test[test.max_floor == 0].index
test.ix[bad_index, "max_floor"] = np.NaN

bad_index = train[train.floor > train.max_floor].index
train.ix[bad_index, "max_floor"] = np.NaN

bad_index = test[test.floor > test.max_floor].index
test.ix[bad_index, "max_floor"] = np.NaN

bad_index = [23584]
train.ix[bad_index, "floor"] = np.NaN

bad_index = train[train.state == 33].index
train.ix[bad_index, "state"] = np.NaN

bad_index = train[train.max_floor >= 57].index
##print(train.ix[bad_index, "max_floor"])
train.ix[bad_index, "max_floor"] = np.NAN
##print(train.max_floor.value_counts())
##print(test.max_floor.value_counts())
##print(train.floor.describe(percentiles= [0.9999]))

##bad_index = train[(train.floor > 1) * (train.max_floor == 1)].index
##print(train.ix[bad_index, ["max_floor","floor"]])
##
##bad_index = test[(test.floor > 1) * (test.max_floor == 1)].index
##print(test.ix[bad_index, ["max_floor","floor"]])

##bad_index = train[train.floor > train.max_floor].index
##print(train.ix[bad_index, ["max_floor","floor"]])
##
##bad_index = test[test.floor > test.max_floor].index
##print(test.ix[bad_index, ["max_floor","floor"]])





test.to_csv("test_clean.csv", index= False, encoding= "utf_8")
train.to_csv("train_clean.csv", index = False, encoding= "utf_8")






print("END OF PROGRAM")
