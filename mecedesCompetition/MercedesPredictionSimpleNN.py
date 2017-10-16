import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

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
##keras.layers.core.Dropout


print("READING TRAIN AND TEST FILES...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("FINISHED READING TRAIN AND TEST FILES!")

##print(train.head())

print("PUTTING TRAIN AND TEST FILES TOGETHER...")
all_data = pd.concat((train.loc[:,'timestamp':'market_count_5000'],
                      test.loc[:,'timestamp':'market_count_5000']))
print("FINISHED PUTTING TRAIN AND TEST FILES TOGETHER!")


print("ANALYZING AND TRANSFORMING DATA FOR BETTER PERFORMANCE...")
#log transform the target:
train["price_doc"] = np.log1p(train["price_doc"])
#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
non_numeric_feats = all_data.dtypes[all_data.dtypes == "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
##print(skewed_feats)
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
print("FINISHED ANALYZING AND TRANSFORMING DATA FOR BETTER PERFORMANCE!")


print("GETTING TRANSFORMED TRAINING DATA....")
X_train = all_data[:train.shape[0]]
##print(train.shape[0])##30471
##print(X_train.shape)##(30471, 1886)
print("GOT TRANSFORMED TRAINING DATA")

print("GETTING TRANSFORMED TEST DATA....")
X_test = all_data[train.shape[0]:]
y = train.price_doc
##print(X_test.shape)##(7662, 1886)
print("GOT TRANSFORMED TEST DATA")

##print(X_train.head())
X_train = StandardScaler().fit_transform(X_train)
##print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
##print(X_train.shape)##nparray
##print(X_train)##(30471, 1886)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 5)

##print(X_tr.shape)##(22853, 1886)
##print(X_val.shape)##(7618, 1886)
##print(y_tr.shape)##(22853,)
##print(y_val.shape)##(7618,)

model = Sequential()
#model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))
##model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))##, kernel_initializer='uniform')

model.add(Dense(1, input_dim=X_train.shape[1], activation='relu',   W_regularizer=l1(0.001))) 

##model.add(BatchNormalization())
##model.add(Activation('tanh'))
##model.add(Dropout(0.2))
##
### we can think of this chunk as the hidden layer    
##model.add(Dense(1, init='uniform'))
##model.add(Dense(1, input_dim=X_train.shape[1], activation='relu')) 
##model.add(BatchNormalization())
####model.add(Activation('tanh'))
##model.add(Dropout(0.2))
##
##### we can think of this chunk as the hidden layer    
##model.add(Dense(1, init='uniform'))
##model.add(BatchNormalization())
####model.add(Activation('tanh'))
##model.add(Dropout(0.2))
##
##### we can think of this chunk as the hidden layer    
##model.add(Dense(1, init='uniform'))
##model.add(BatchNormalization())
####model.add(Activation('tanh'))
##model.add(Dropout(0.2))
##
##### we can think of this chunk as the hidden layer    
##model.add(Dense(1, init='uniform'))
##model.add(BatchNormalization())
####model.add(Activation('tanh'))
##model.add(Dropout(0.2))

##
### we can think of this chunk as the output layer
##model.add(Dense(2, init='uniform'))
##model.add(BatchNormalization())
##model.add(Activation('softmax'))
##
### setting up the optimization of our weights 
##sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
##model.compile(loss='binary_crossentropy', optimizer=sgd)




##model.compile(loss = "mse", optimizer = sgd)
model.compile(loss = "mse", optimizer = "adam")



model.summary()

##hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))

##model.fit(X_tr, y_tr, validation_data = (X_val, y_val), epochs=20, batch_size=16,validation_split=0.2, verbose = 2)
model.fit(X_tr, y_tr, shuffle=True, epochs=1, batch_size=11, validation_data = (X_val, y_val))##, shuffle=True, epochs=100, batch_size=16
##print(model.predict(X_val).shape)
##print(model.predict(X_val))
pd.Series(model.predict(X_val)[:,0]).hist()
plt.show()





