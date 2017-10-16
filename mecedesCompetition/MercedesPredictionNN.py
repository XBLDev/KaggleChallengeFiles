import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
##from sklearn.model_selection import cross_val_score

import xgboost as xgb

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout


from keras import optimizers


import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

print("READING TRAIN AND TEST FILES...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("FINISHED READING TRAIN AND TEST FILES!")

print("PUTTING TRAIN AND TEST FILES TOGETHER...")
all_data = pd.concat((train.loc[:,'X0':'X385'],
                      test.loc[:,'X0':'X385']))
print("FINISHED PUTTING TRAIN AND TEST FILES TOGETHER!")





print("ANALYZING AND TRANSFORMING DATA FOR BETTER PERFORMANCE...")
train["y"] = np.log1p(train["y"])
all_data = pd.get_dummies(all_data)
print("FINISHED ANALYZING AND TRANSFORMING DATA FOR BETTER PERFORMANCE!")


print("GETTING TRANSFORMED TRAINING DATA....")
X_train = all_data[:train.shape[0]]
y = train.y
print("GOT TRANSFORMED TRAINING DATA")

print("GETTING TRANSFORMED TEST DATA....")
X_test = all_data[train.shape[0]:]
print(X_test.shape)##(7662, 1886)
print("GOT TRANSFORMED TEST DATA")


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features, [-1, 7662 * 1886])
  
 


##X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 5)
##model = Sequential()
##model.add(Dense(1, input_dim= X_train.shape[1], activation='relu',   W_regularizer=l1(0.001))) 
##model.compile(loss = "mse", optimizer = "adam")
##model.summary()
##
##model.fit(X_tr, y_tr, shuffle=True, epochs=10000, batch_size=16, validation_data = (X_val, y_val))##, shuffle=True, epochs=100, batch_size=16 batch_size=11
##result = model.predict(X_test.values)
##result = np.expm1(result)



#applying log transformation
##train["y"] = np.log1p(train["y"])

##y = np.log(y)
##sns.distplot(y, fit=norm);
##fig = plt.figure()
##res = stats.probplot(y, plot=plt)
##plt.show()
##print("Skewness: %f" % y.skew())##0.395163
##print("Kurtosis: %f" % y.kurt())##7.910713

##log1p:
##Skewness: 0.395163
##Kurtosis: 1.326792
##log:
##Skewness: 0.389980
##Kurtosis: 1.309540





print("END OF PROGRAM")
