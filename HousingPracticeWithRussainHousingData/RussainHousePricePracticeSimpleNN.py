import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

##from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
##from sklearn.model_selection import cross_val_score

##import xgboost as xgb


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


##print(np.expm1(235119))

print("READING TRAIN AND TEST FILES...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("FINISHED READING TRAIN AND TEST FILES!")

##print(train.head())

print("PUTTING TRAIN AND TEST FILES TOGETHER...")
all_data = pd.concat((train.loc[:,'timestamp':'market_count_5000'],
                      test.loc[:,'timestamp':'market_count_5000']))
print("FINISHED PUTTING TRAIN AND TEST FILES TOGETHER!")
##print(all_data["life_sq"])
##print(all_data["life_sq"].values.shape)##(38133,)
##print(all_data)##[38133 rows x 290 columns]



print("ANALYZING AND TRANSFORMING DATA FOR BETTER PERFORMANCE...")
#log transform the target:

price_doc_withoutLog1p = train["price_doc"]
##print(price_doc_withoutLog1p.shape)
##train["price_doc"] = np.log1p(train["price_doc"])
#log transform skewed numeric features:
##numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
##non_numeric_feats = all_data.dtypes[all_data.dtypes == "object"].index
##skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
##skewed_feats = skewed_feats[skewed_feats > 0.75]
##skewed_feats = skewed_feats.index
##all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
##print(all_data["timestamp"])##NOTE: TIMESTAMP DISAPPEARS AFTER GET_DUMMIES
all_data = all_data.fillna(all_data.mean())
##print(all_data)##[38133 rows x 1886 columns]

##print(all_data["life_sq"])
##print(all_data["life_sq"].values.shape)##(38133,)

##print(type(all_data["full_sq"]))##<class 'pandas.core.series.Series'>

print("FINISHED ANALYZING AND TRANSFORMING DATA FOR BETTER PERFORMANCE!")

####train["price_doc"]
print("GETTING TRANSFORMED TRAINING DATA....")
X_train = all_data[:train.shape[0]]
y = train.price_doc
##print(y)
##print(y.shape)
##print(type(X_train))##<class 'numpy.ndarray'>
##print(train.shape[0])##30471
##print(X_train.shape)##(30471, 1886)
print("GOT TRANSFORMED TRAINING DATA")

print("GETTING TRANSFORMED TEST DATA....")
X_test = all_data[train.shape[0]:]
##print(X_test.shape)##(7662, 1886)
print("GOT TRANSFORMED TEST DATA")
##
##
##print(X_train)
##print(type(X_train))##<class 'pandas.core.frame.DataFrame'>
##print(X_train.shape)
X_train = StandardScaler().fit_transform(X_train)
##print(y)
##print(np.mean(X_train, axis=0))
##print(np.mean(X_train, axis=1))
##print(X_train)
##print(type(X_train))##<class 'numpy.ndarray'>

##print(type(X_train))##<class 'numpy.ndarray'>
##print(X_train)
##print(X_train.shape)
##print(X_train.head())
##print(X_test.shape)##(7662, 1886)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 5)

##print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
##print(X_train.shape)##nparray
##print(X_train)##(30471, 1886)
##print(X_train.shape[1])

##print(X_tr.shape)##(22853, 1886)
##print(X_val.shape)##(7618, 1886)
##print(y_tr.shape)##(22853,)
##print(y_val.shape)##(7618,)

model = Sequential()
###model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))
####model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))##, kernel_initializer='uniform')
model.add(Dense(1, input_dim= X_train.shape[1], activation='relu',   W_regularizer=l1(0.001))) 
##
####model.add(Dense(1, input_dim=[None, X_train.shape[1]], activation='relu',   W_regularizer=l1(0.001))) 
##
##model.add(BatchNormalization())
####model.add(Activation('tanh'))
##model.add(Dropout(0.2))
####
##### we can think of this chunk as the hidden layer    
##model.add(Dense( 1, init='uniform', activation='relu', W_regularizer=l1(0.001)))

##ValueError: Error when checking model target: expected batch_normalization_2 to have shape (None, 2)
##but got array with shape (22853, 1)

####model.add(Dense(1, input_dim=X_train.shape[1], activation='relu')) 
##model.add(BatchNormalization())
######model.add(Activation('tanh'))
##model.add(Dropout(0.2))
####
####### we can think of this chunk as the hidden layer    
##model.add(Dense(1, init='uniform', activation='relu', W_regularizer=l1(0.001)))
##model.add(BatchNormalization())
##########model.add(Activation('tanh'))
##model.add(Dropout(0.2))
##
##
######
######### we can think of this chunk as the hidden layer    
##model.add(Dense(1, init='uniform'))
##model.add(BatchNormalization())
########model.add(Activation('tanh'))
##model.add(Dropout(0.2))
######
######### we can think of this chunk as the hidden layer    
##model.add(Dense(1, init='uniform'))
##model.add(BatchNormalization())
########model.add(Activation('tanh'))
##model.add(Dropout(0.2))
##
####
##### we can think of this chunk as the output layer
####model.add(Dense(2, init='uniform'))
####model.add(BatchNormalization())
####model.add(Activation('softmax'))
####
##### setting up the optimization of our weights 
####sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
####model.compile(loss='binary_crossentropy', optimizer=sgd)
##
##
##
##
####model.compile(loss = "mse", optimizer = sgd)
model.compile(loss = "mse", optimizer = "adam")
##
##
##
model.summary()
##
##
####print(X_train.head())
##
####print(X_val)
####print(X_val.shape)##(7618, 1886)
##
####hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))
##
####model.fit(X_tr, y_tr, validation_data = (X_val, y_val), epochs=20, batch_size=16,validation_split=0.2, verbose = 2)
model.fit(X_tr, y_tr, shuffle=True, epochs=10000, batch_size=16, validation_data = (X_val, y_val))##, shuffle=True, epochs=100, batch_size=16 batch_size=11
####print(model.predict(X_val).shape)##(7618, 1)
####print(type(model.predict(X_val)))##<class 'numpy.ndarray'>
##
####print(model.predict(X_val)[:,0])
####print(model.predict(X_val)[:,0].shape)##(7618,)
####pd.Series(model.predict(X_val)[:,0]).hist()
####plt.show()
####np.reshape(a, (3,-1))
##
####shapeOFXVal = X_val.shape
##
####print(X_test)##(7662, 1886)[7662 rows x 1886 columns]
####X_test = np.reshape(X_test,(7662, 1886))
####X_test = np.reshape(X_test, shapeOFXVal)
result = model.predict(X_test.values)
####print(type(result))##<class 'numpy.ndarray'>
####print(result.shape)##(7662, 1)
####print(result[0])##[ 38761.78125]
####print(result[0].shape)##(1,)
####print(result[1])##(1,)
####print(result[1].shape)##[ 911.47216797]
####print(result[:1])##[ 38761.78125]
####print(result[:1].shape)##(1,)
####print(result)
result = np.reshape(result,(7662,))
####result = result * 100
result = np.around(result)

####print(result)
solution = pd.DataFrame({"id": test.id, "price_doc": result})
########ValueError: Cannot feed value of shape (7662, 32) for Tensor 'dense_1_input:0', which has shape '(?, 1886)'
solution.to_csv("RussianHousingPredictNN.csv", index = False)
##
##y_new_inverse = scalery.inverse_transform(y_new)
