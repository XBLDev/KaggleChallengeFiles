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

macro_df = pd.read_csv("macro.csv", parse_dates=['timestamp'])

###missing data
##total = macro_df.isnull().sum().sort_values(ascending=False)
##percent = (macro_df.isnull().sum()/macro_df.isnull().count()).sort_values(ascending=False)
##missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
##print(missing_data[missing_data['Total'] > 1])
##print(missing_data)
####We'll consider that when more than 15% of the data is missing,
####we should delete the corresponding variable and pretend it never existed. 
##print(missing_data.head(20))

macro_df_corrmat = macro_df.corr()
k = 10 #number of variables for heatmap
cols = macro_df_corrmat.nlargest(k, 'salary')['salary'].index
cm = np.corrcoef(macro_df_corrmat[cols].values.T)
##cm = np.corrcoef(train_data_corrmat[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
sns.plt.show()


print("END OF PROGRAM")
