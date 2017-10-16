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

train_df = pd.read_csv("train_clean.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("test_clean.csv", parse_dates=['timestamp'])

all_data = pd.concat((train_df.loc[:,'timestamp':'market_count_5000'],
                      test_df.loc[:,'timestamp':'market_count_5000']))


train_df_fillNA = train_df.fillna(train_df.median())
train_df_without_NAN = train_df.dropna()
test_df_without_NAN = test_df.dropna()
all_data_without_NAN = all_data.dropna()

train_df_without_NAN['product_type'] = train_df_without_NAN['product_type'].fillna(train_df_without_NAN['product_type'].value_counts().index[0])
all_data['product_type'] = all_data['product_type'].fillna(all_data['product_type'].value_counts().index[0])

all_data.ecology[all_data.ecology == 'excellent'] = 4
all_data.ecology[all_data.ecology == 'good'] = 3
all_data.ecology[all_data.ecology == 'satisfactory'] = 2
all_data.ecology[all_data.ecology == 'poor'] = 1
all_data.ecology[all_data.ecology == 'no data'] = 0
all_data.ecology = all_data.ecology.astype('int')
all_data.ecology[all_data.ecology == 0] = all_data.ecology.median()

train_df_without_NAN.ecology[train_df_without_NAN.ecology == 'excellent'] = 4
train_df_without_NAN.ecology[train_df_without_NAN.ecology == 'good'] = 3
train_df_without_NAN.ecology[train_df_without_NAN.ecology == 'satisfactory'] = 2
train_df_without_NAN.ecology[train_df_without_NAN.ecology == 'poor'] = 1
train_df_without_NAN.ecology[train_df_without_NAN.ecology == 'no data'] = 0
train_df_without_NAN.ecology = train_df_without_NAN.ecology.astype('int')
train_df_without_NAN.ecology[train_df_without_NAN.ecology == 0] = train_df_without_NAN.ecology.median()


train_df_without_NAN["yearmonth"] = train_df_without_NAN["timestamp"].dt.year*100 + train_df_without_NAN["timestamp"].dt.month

train_df_without_NAN["yearweek"] = train_df_without_NAN["timestamp"].dt.year*100 + train_df_without_NAN["timestamp"].dt.weekofyear

train_df_without_NAN["year"] = train_df_without_NAN["timestamp"].dt.year


###Price of the house could also be affected by the availability of other houses at the same time period.
###So creating a count variable on the number of houses at the given time period might help.
def add_count(df, group_col):
    grouped_df = df.groupby(group_col)["id"].aggregate("count").reset_index()
    grouped_df.columns = [group_col, "count_"+group_col]
    print(grouped_df.columns)
    df = pd.merge(df, grouped_df, on=group_col, how="left")
    return df

train_df_without_NAN = add_count(train_df_without_NAN, "yearmonth")
##print(train_df_without_NAN.columns)

yearmonth_cols = [col for col in train_df_without_NAN.columns if 'yearmonth' in col]##<------it drops the rmse, but not much
print(yearmonth_cols)

all_data["year"] = all_data["timestamp"].dt.year
all_data_without_NAN["year"] = all_data_without_NAN["timestamp"].dt.year
train_df_without_NAN["year"] = train_df_without_NAN["timestamp"].dt.year

train_df_without_NAN["age_of_building"] = train_df_without_NAN["build_year"] - train_df_without_NAN["year"]
all_data["age_of_building"] = all_data["build_year"] - all_data["year"]

all_data["life_and_kitchen"] = all_data["kitch_sq"] + all_data["life_sq"]
test_df["life_and_kitchen"] = test_df["kitch_sq"] + test_df["life_sq"]
train_df_without_NAN["life_and_kitchen"] = train_df_without_NAN["kitch_sq"] + train_df_without_NAN["life_sq"]

all_data["full_minus_life_and_kitchen"] = all_data["full_sq"] - all_data["life_sq"] - all_data["kitch_sq"]
test_df["full_minus_life_and_kitchen"] = test_df["full_sq"] - test_df["life_sq"] - test_df["kitch_sq"]
train_df_without_NAN["full_minus_life_and_kitchen"] = train_df_without_NAN["full_sq"] - train_df_without_NAN["life_sq"] - train_df_without_NAN["kitch_sq"]

all_data["full_sq_times_state"] = all_data["full_sq"] * all_data["state"]
test_df["full_sq_times_state"] = test_df["full_sq"] * test_df["state"]
train_df_without_NAN["full_sq_times_state"] = train_df_without_NAN["full_sq"] * train_df_without_NAN["state"]

all_data["num_room_times_state"] = all_data["num_room"] * all_data["state"]
test_df["num_room_times_state"] = test_df["num_room"] * test_df["state"]
train_df_without_NAN["num_room_times_state"] = train_df_without_NAN["num_room"] * train_df_without_NAN["state"]

all_data["life_and_kitchen_times_state"] = (all_data["kitch_sq"] + all_data["life_sq"]) * all_data["state"]
test_df["life_and_kitchen_times_state"] = (test_df["kitch_sq"] + test_df["life_sq"])* test_df["state"]
train_df_without_NAN["life_and_kitchen_times_state"] = (train_df_without_NAN["kitch_sq"] + train_df_without_NAN["life_sq"])* train_df_without_NAN["state"]




train_df_without_NAN["ratio_floor_max_floor"] = train_df_without_NAN["floor"] / train_df_without_NAN["max_floor"].astype("float")
train_df_without_NAN["floor_from_top"] = train_df_without_NAN["max_floor"] - train_df_without_NAN["floor"]

train_df_without_NAN["ratio_life_sq_full_sq"] = train_df_without_NAN["life_sq"] / np.maximum(train_df_without_NAN["full_sq"].astype("float"),1)
train_df_without_NAN["ratio_life_sq_full_sq"].ix[train_df_without_NAN["ratio_life_sq_full_sq"]<0] = 0
train_df_without_NAN["ratio_life_sq_full_sq"].ix[train_df_without_NAN["ratio_life_sq_full_sq"]>1] = 1




# month of year #
train_df["month_of_year"] = train_df["timestamp"].dt.month
test_df["month_of_year"] = test_df["timestamp"].dt.month
train_df_without_NAN["month_of_year"] = train_df_without_NAN["timestamp"].dt.month

# week of year #
train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear
test_df["week_of_year"] = test_df["timestamp"].dt.weekofyear
train_df_without_NAN["week_of_year"] = train_df_without_NAN["timestamp"].dt.weekofyear

# day of week #
train_df["day_of_week"] = train_df["timestamp"].dt.weekday
train_df_without_NAN["day_of_week"] = train_df_without_NAN["timestamp"].dt.weekday


# ratio of kitchen area to living area #
train_df["ratio_kitch_sq_life_sq"] = train_df["kitch_sq"] / np.maximum(train_df["life_sq"].astype("float"),1)
test_df["ratio_kitch_sq_life_sq"] = test_df["kitch_sq"] / np.maximum(test_df["life_sq"].astype("float"),1)
train_df_without_NAN["ratio_kitch_sq_life_sq"] = train_df_without_NAN["kitch_sq"] / np.maximum(train_df_without_NAN["life_sq"].astype("float"),1)
train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]<0] = 0
train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]>1] = 1
test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]<0] = 0
test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]>1] = 1
train_df_without_NAN["ratio_kitch_sq_life_sq"].ix[train_df_without_NAN["ratio_kitch_sq_life_sq"]<0] = 0
train_df_without_NAN["ratio_kitch_sq_life_sq"].ix[train_df_without_NAN["ratio_kitch_sq_life_sq"]>1] = 1

# ratio of kitchen area to full area #
train_df["ratio_kitch_sq_full_sq"] = train_df["kitch_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)
test_df["ratio_kitch_sq_full_sq"] = test_df["kitch_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)
train_df_without_NAN["ratio_kitch_sq_full_sq"] = train_df_without_NAN["kitch_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)
train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]<0] = 0
train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]>1] = 1
test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]<0] = 0
test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]>1] = 1
train_df_without_NAN["ratio_kitch_sq_full_sq"].ix[train_df_without_NAN["ratio_kitch_sq_full_sq"]<0] = 0
train_df_without_NAN["ratio_kitch_sq_full_sq"].ix[train_df_without_NAN["ratio_kitch_sq_full_sq"]>1] = 1

# full_sq - life_sq #
train_df_without_NAN["extra_sq"] = train_df_without_NAN["full_sq"] - train_df_without_NAN["life_sq"]

# extra ratio to full #

train_df_without_NAN["extra_sq_to_full"] =  (train_df_without_NAN["full_sq"] - train_df_without_NAN["life_sq"])/train_df_without_NAN["full_sq"]
all_data["extra_sq_to_full"] =  (all_data["full_sq"] - all_data["life_sq"])/all_data["full_sq"]

# age of building #
train_df_without_NAN["age_of_building"] = train_df_without_NAN["build_year"] - train_df_without_NAN["year"]



train_df_without_NAN["cafe_count_5000_allPriceRange_sum"] = train_df_without_NAN["cafe_count_5000_price_500"] + train_df_without_NAN["cafe_count_5000_price_1000"] + train_df_without_NAN["cafe_count_5000_price_1500"] +train_df_without_NAN["cafe_count_5000_price_2500"] + train_df_without_NAN["cafe_count_5000_price_4000"] + train_df_without_NAN["cafe_count_5000_price_high"]   
all_data["cafe_count_5000_allPriceRange_sum"] = all_data["cafe_count_5000_price_500"] + all_data["cafe_count_5000_price_1000"] + all_data["cafe_count_5000_price_1500"] + all_data["cafe_count_5000_price_2500"] + all_data["cafe_count_5000_price_4000"] + all_data["cafe_count_5000_price_high"]   

train_df_without_NAN["full_life_kitch"] = train_df_without_NAN["full_sq"] + train_df_without_NAN["life_sq"] + train_df_without_NAN["kitch_sq"]
all_data["full_life_kitch"] = all_data["full_sq"] + all_data["life_sq"] + all_data["kitch_sq"]

train_df_without_NAN["full_life"] = train_df_without_NAN["full_sq"] + train_df_without_NAN["life_sq"]
all_data["full_life"] = all_data["full_sq"] + all_data["life_sq"]

train_df_without_NAN["ratio_full_floor"] = train_df_without_NAN["full_sq"] / train_df_without_NAN["floor"]
all_data["ratio_full_floor"] = all_data["full_sq"] / all_data["floor"]

train_df_without_NAN["area_per_room"] = train_df_without_NAN["life_sq"] / train_df_without_NAN["num_room"]
all_data["area_per_room"] = all_data["life_sq"] / all_data["num_room"]

train_df_without_NAN["full_minus_lifeAndKitch"] = train_df_without_NAN["full_sq"] - train_df_without_NAN["life_sq"] - train_df_without_NAN["kitch_sq"]
all_data["full_minus_lifeAndKitch"] = all_data["full_sq"] - all_data["life_sq"] - all_data["kitch_sq"]

train_df_without_NAN["full_minus_lifeAndKitch_to_full"] = (train_df_without_NAN["full_sq"] - train_df_without_NAN["life_sq"] - train_df_without_NAN["kitch_sq"])/train_df_without_NAN["full_sq"]
all_data["full_minus_lifeAndKitch_to_full"] = (all_data["full_sq"] - all_data["life_sq"] - all_data["kitch_sq"])/all_data["full_sq"]

train_df_without_NAN["full_to_buildingAge"] = train_df_without_NAN["full_sq"] / (train_df_without_NAN["build_year"] - train_df_without_NAN["timestamp"].dt.year).astype("float")
all_data["full_to_buildingAge"] = all_data["full_sq"] / (all_data["build_year"] - all_data["timestamp"].dt.year).astype("float")



##**************************************************************cafe FEATURES*************************************************************

train_df_without_NAN["cafe_notNA_price_500"] = train_df_without_NAN["cafe_count_500"] - train_df_without_NAN["cafe_count_500_na_price"]
all_data["cafe_notNA_price_500"] = all_data["cafe_count_500"] - all_data["cafe_count_500_na_price"]

train_df_without_NAN["cafe_notNA_price_1000"] = train_df_without_NAN["cafe_count_1000"] - train_df_without_NAN["cafe_count_1000_na_price"]
all_data["cafe_notNA_price_1000"] = all_data["cafe_count_1000"] - all_data["cafe_count_1000_na_price"]

train_df_without_NAN["cafe_notNA_price_1500"] = train_df_without_NAN["cafe_count_1500"] - train_df_without_NAN["cafe_count_1500_na_price"]
all_data["cafe_notNA_price_1500"] = all_data["cafe_count_1500"] - all_data["cafe_count_1500_na_price"]

train_df_without_NAN["cafe_notNA_price_2000"] = train_df_without_NAN["cafe_count_2000"] - train_df_without_NAN["cafe_count_2000_na_price"]
all_data["cafe_notNA_price_2000"] = all_data["cafe_count_2000"] - all_data["cafe_count_2000_na_price"]

train_df_without_NAN["cafe_notNA_price_3000"] = train_df_without_NAN["cafe_count_3000"] - train_df_without_NAN["cafe_count_3000_na_price"]
all_data["cafe_notNA_price_3000"] = all_data["cafe_count_3000"] - all_data["cafe_count_3000_na_price"]

train_df_without_NAN["cafe_notNA_price_5000"] = train_df_without_NAN["cafe_count_5000"] - train_df_without_NAN["cafe_count_5000_na_price"]
all_data["cafe_notNA_price_5000"] = all_data["cafe_count_5000"] - all_data["cafe_count_5000_na_price"]

##**************************************************************cafe FEATURES*************************************************************

##**************************************************************entertainment FEATURES*************************************************************

all_data["entertainment_Count_500"] = all_data["trc_count_500"] + all_data["cafe_count_500"]+ all_data["sport_count_500"]
test_df["entertainment_Count_500"] = test_df["trc_count_500"] + test_df["cafe_count_500"]+ test_df["sport_count_500"]

all_data["entertainment_Count_5000"] = all_data["trc_count_5000"] + all_data["cafe_count_5000"]+ all_data["sport_count_5000"]
test_df["entertainment_Count_5000"] = test_df["trc_count_5000"] + test_df["cafe_count_5000"]+ test_df["sport_count_5000"]
train_df_without_NAN["entertainment_Count_5000"] = train_df_without_NAN["trc_count_5000"] + train_df_without_NAN["cafe_count_5000"]+ train_df_without_NAN["sport_count_5000"]

##**************************************************************entertainment FEATURES*************************************************************

##**************************************************************EDUCATION FEATURES*************************************************************

all_data["school_count_total"] = all_data["preschool_education_centers_raion"] + all_data["school_education_centers_raion"] + all_data["university_top_20_raion"] + all_data["additional_education_raion"]
train_df_without_NAN["school_count_total"] = train_df_without_NAN["preschool_education_centers_raion"] + train_df_without_NAN["school_education_centers_raion"] + train_df_without_NAN["university_top_20_raion"] + train_df_without_NAN["additional_education_raion"]

all_data["preschool_seat_available_rate"] = all_data["children_preschool"] - all_data["preschool_quota"]
train_df_without_NAN["preschool_seat_available_rate"] = train_df_without_NAN["children_preschool"] - train_df_without_NAN["preschool_quota"]

# ratio of children_preschool and preschool_quota #
train_df_without_NAN["ratio_preschool"] = train_df_without_NAN["children_preschool"] / train_df_without_NAN["preschool_quota"].astype("float")

# ratio of children_school and school_quota #
train_df_without_NAN["ratio_school"] = train_df_without_NAN["children_school"] / train_df_without_NAN["school_quota"].astype("float")

train_df_without_NAN["top20Highschool_to_numberOfHighSchool"] = train_df_without_NAN["school_education_centers_top_20_raion"] / train_df_without_NAN["school_education_centers_raion"]
all_data["top20Highschool_to_numberOfHighSchool"] = all_data["school_education_centers_top_20_raion"] / all_data["school_education_centers_raion"]

##**************************************************************EDUCATION FEATURES*************************************************************

print("Correlation between price_doc and other columns, hightest is full_sq: 0.74: ")
##

##print("life_and_kitchen ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['life_and_kitchen']))##0.686770380517
##print("full_sq_times_state ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['full_sq_times_state']))##0.642629934279
##print("life_and_kitchen_times_state ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['life_and_kitchen_times_state']))##0.622451819926
##print("num_room_times_state ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['num_room_times_state']))##0.474193707785
##print("entertainment_Count_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['entertainment_Count_5000']))##0.304184681686
##print("full_minus_life_and_kitchen ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['full_minus_life_and_kitchen']))##0.547194303574
##print("school_count_total ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['school_count_total']))##0.0791496447021
##print("preschool_seat_available_rate ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['preschool_seat_available_rate']))##-0.00453697263668
##print("cafe_count_5000_allPriceRange_sum ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_5000_allPriceRange_sum']))##0.304664957922
##print("full_life_kitch ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['full_life_kitch']))## 0.721706575901
##print("full_life ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['full_life']))## 0.714106697544
##print("ratio_full_floor ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['ratio_full_floor']))## 0.126241647571
##print("top20Highschool_to_numberOfHighSchool ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['top20Highschool_to_numberOfHighSchool']))## 0.0901428778236241647571
##print("area_per_room ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['area_per_room']))## 0.178103287616
##print("count_yearmonth ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['count_yearmonth']))## -0.0107299013202
##print("full_minus_lifeAndKitch ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['full_minus_lifeAndKitch']))##0.547194303574
##print("full_minus_lifeAndKitch_to_full ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['full_minus_lifeAndKitch_to_full']))##0.0656248499906
##print("cafe_notNA_price_500 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_notNA_price_500']))##0.27806967303
##print("cafe_notNA_price_1000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_notNA_price_1000']))##0.277616407629
##print("cafe_notNA_price_1500 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_notNA_price_1500']))##0.28497368669
##print("cafe_notNA_price_2000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_notNA_price_2000']))##0.285502061434
##print("cafe_notNA_price_3000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_notNA_price_3000']))##0.300208568031
##print("cafe_notNA_price_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_notNA_price_5000']))##0.304664957922
##print("full_to_buildingAge ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['full_to_buildingAge']))##nan

##print("ratio_floor_max_floor ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['ratio_floor_max_floor']))##0.0130794984932
##print("floor_from_top ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['floor_from_top']))##0.12177908001
##print("ratio_life_sq_full_sq ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['ratio_life_sq_full_sq']))##0.0620511242135
##print("yearmonth ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['yearmonth']))##0.087757866761
##print("yearweek ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['yearweek']))##0.0876573546376
##print("year ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['year']))##0.0870070631995
##print("month_of_year ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['month_of_year']))##-0.0175805141567
##print("week_of_year ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['week_of_year']))##-0.0177050612112
##print("day_of_year ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['day_of_week']))## -0.0114275139566
##print("ratio_kitch_sq_life_sq ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['ratio_kitch_sq_life_sq']))##0.0620511242135
##print("ratio_kitch_sq_full_sq ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['ratio_kitch_sq_full_sq']))##nan
##
##print("extra_sq ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['extra_sq']))##0.606603459986
##print("age_of_building ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['age_of_building']))##0.129669102624
##print("ratio_preschool ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['ratio_preschool']))##nan
##print("ratio_school ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['ratio_school']))##-0.110133808866


##print("full_sq ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['full_sq']))## 0.731484416742
##print("life_sq ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['life_sq']))## 0.655698575341
##print("kitch_sq ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['kitch_sq']))## 0.518723198533
##print("num_room ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['num_room']))## 0.499181677345
##print("build_year ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['build_year']))## 0.132332321114
##
##print("floor ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['floor']))## 0.171316063466
##print("max_floor ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['max_floor']))## 0.232672810506
##print("material ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['material']))## 0.0826905673257
##print("state ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['state']))## 0.0988511517888
##
##
##print("cafe_count_500 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_500']))##0.274791251899
##print("cafe_count_1000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_1000']))## IT GETS BIGGER FROM 500 TO 5000
##print("cafe_count_1500 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_1500']))##
##print("cafe_count_2000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_2000']))##
##print("cafe_count_3000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_3000']))##
##print("cafe_count_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_5000']))##0.30427176007
##
##print("cafe_avg_price_500 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_avg_price_500']))##0.089
##print("cafe_avg_price_1000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_avg_price_1000']))## IT GETS BIGGER FROM 500 TO 5000
##print("cafe_avg_price_1500 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_avg_price_1500']))##
##print("cafe_avg_price_2000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_avg_price_2000']))##
##print("cafe_avg_price_3000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_avg_price_3000']))##
##print("cafe_avg_price_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_avg_price_5000']))##0.24
##
##print("", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_500_na_price']))##0.186488376261
##print("", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_1000_na_price']))## IT GETS BIGGER FROM 500 TO 5000
##print("", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_1500_na_price']))##
##print("", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_2000_na_price']))##
##print("", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_3000_na_price']))##
##print("", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['cafe_count_5000_na_price']))##0.29482940257
##
##



##
##print("", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['preschool_quota']))##-0.131648496978
##print("", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['school_quota']))##-0.000351525867912
##
##print("", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['university_top_20_raion']))##0.2016078825
##print("preschool_education_centers_raion ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['preschool_education_centers_raion']))##0.0152755877865
##print("school_education_centers_raion ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['school_education_centers_raion']))##0.0415103092059
##print("additional_education_raion ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['additional_education_raion']))##0.0594854894866
##print("children_preschool ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['children_preschool']))##-0.0843556662212
##print("preschool_quota ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['preschool_quota']))##-0.131648496978
##print("school_education_centers_top_20_raion ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['school_education_centers_top_20_raion']))##0.146014759044
##print("school_education_centers_raion ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['school_education_centers_raion']))##0.0415103092059
##print("university_top_20_raion ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['university_top_20_raion']))##0.2016078825
##print("ecology ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['ecology']))## 0.0541968913296
##print("green_part_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['green_part_5000']))## -0.154079484858
##print("prom_part_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['prom_part_5000']))## -0.121640942165
##print("office_count_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['office_count_5000']))##0.28743237885
##print("office_sqm_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['office_sqm_5000']))## 0.300905259337
##print("trc_count_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['trc_count_5000']))## 0.255839817888
##print("trc_sqm_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['trc_sqm_5000']))## 0.203753548602
##print("big_church_count_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['big_church_count_5000']))## 0.257906179498
##print("mosque_count_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['mosque_count_5000']))## 0.245808786172
##print("leisure_count_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['leisure_count_5000']))## 0.278687787067
##print("sport_count_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['sport_count_5000']))## 0.278757301583
##print("market_count_5000 ", train_df_without_NAN['price_doc'].corr(train_df_without_NAN['market_count_5000']))## 0.0603560874003




































##train_df_fillNA_corrmat = train_df_fillNA.corr()
##k = 10 #number of variables for heatmap
##cols = train_df_fillNA_corrmat.nlargest(k, 'price_doc')['price_doc'].index
##cm = np.corrcoef(train_df_fillNA[cols].values.T)
####cm = np.corrcoef(train_data_corrmat[cols].values.T)
##sns.set(font_scale=1)
##hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
##plt.yticks(rotation=0)
##plt.xticks(rotation=90)
##plt.show()
##sns.plt.show()

##train_df_corrmat = train_df.corr()
##k = 10 #number of variables for heatmap
##cols = train_df_corrmat.nlargest(k, 'price_doc')['price_doc'].index
##cm = np.corrcoef(train_df[cols].values.T)
####cm = np.corrcoef(train_data_corrmat[cols].values.T)
##sns.set(font_scale=1)
##hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
##plt.yticks(rotation=0)
##plt.xticks(rotation=90)
##plt.show()
##sns.plt.show()


##train_df_without_NAN_corrmat = train_df_without_NAN.corr()
##k = 15 #number of variables for heatmap
##cols = train_df_without_NAN_corrmat.nlargest(k, 'price_doc')['price_doc'].index
##print(cols)
##cm = np.corrcoef(train_df_without_NAN[cols].values.T)
####cm = np.corrcoef(train_data_corrmat[cols].values.T)
##sns.set(font_scale=1)
##hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
##plt.yticks(rotation=0)
##plt.xticks(rotation=90)
##plt.show()
##sns.plt.show()

##
##train_df_without_NaN_corrmat = train_df_without_NaN.corr()
##k = 10 #number of variables for heatmap
##cols = train_df_without_NaN_corrmat.nsmallest(k, 'price_doc')['price_doc'].index
##cm = np.corrcoef(train_df_without_NaN[cols].values.T)
####cm = np.corrcoef(train_data_corrmat[cols].values.T)
##sns.set(font_scale=1.25)
##hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
##plt.yticks(rotation=0)
##plt.xticks(rotation=90)
##plt.show()
##sns.plt.show()



print("END OF PROGRAM")
