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
from tensorflow.python.client import device_lib

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())



train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

all_data = pd.concat((train.loc[:,'X0':'X385'],
                      test.loc[:,'X0':'X385']))


all_data = pd.get_dummies(all_data)
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.y

testIDs = test.ID
print(np.vstack((testIDs.values,testIDs.values)).T)

X_train = X_train.values
X_test = X_test.values

y = y.values






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

##print(X_train.shape)##(4209, 579)
##print(type(X_train))##<class 'numpy.ndarray'>
##print(y.shape)##(4209,)
##print(type(y))##<class 'numpy.ndarray'>
##print(10, X_train, y)






##BELOW IS THE TENSOR FLOW PART OF THIS PROGRAM
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

keep_prob_fc1 = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, shape=[None, 579])##holder for training data
y_ = tf.placeholder(tf.float32, shape=[None, 1])##holder for truth values

W_fc1 = weight_variable([579,579])
b_fc1 = bias_variable([579])

##y_output = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
fc1 = tf.nn.dropout(fc1, 0.8)
##print(y_output.get_shape())##(?, 64)
##print(fc1.get_shape())##(?, 64)

W_fc2 = weight_variable([579, 579])
b_fc2 = bias_variable([579])

fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)
fc2 = tf.nn.dropout(fc2, 0.7)
##
W_fc3 = weight_variable([579,579])
b_fc3 = bias_variable([579])
##
fc3 = tf.nn.relu(tf.matmul(fc2, W_fc3) + b_fc3)
fc3 = tf.nn.dropout(fc3, 0.7)
##
W_fc4 = weight_variable([579,579])
b_fc4 = bias_variable([579])
##
fc4 = tf.nn.relu(tf.matmul(fc3, W_fc4) + b_fc4)
fc4 = tf.nn.dropout(fc4, 0.7)
##
W_fc5 = weight_variable([579 , 579])
b_fc5 = bias_variable([579])

fc5 = tf.nn.relu(tf.matmul(fc4, W_fc5) + b_fc5)
fc5 = tf.nn.dropout(fc5, 0.7)
##
W_fc6 = weight_variable([579, 1])
b_fc6 = bias_variable([1])

y_output = tf.nn.relu(tf.matmul(fc5, W_fc6) + b_fc6)


##print(y_output.get_shape())##(?, 1)

##total_error = tf.reduce_sum(tf.square(tf.sub(y, tf.reduce_mean(y))))
##unexplained_error = tf.reduce_sum(tf.square(tf.sub(y, prediction)))
##R_squared = tf.sub(1, tf.div(total_error, unexplained_error))

##total_error = tf.reduce_sum(tf.square(tf.sub(y, tf.reduce_mean(y))))
##unexplained_error = tf.reduce_sum(tf.square(tf.sub(y, prediction)))
##R_squared = tf.sub(tf.div(total_error, unexplained_error),1.0)

total_error = tf.reduce_sum(tf.square(tf.subtract(y_, tf.reduce_mean(y_))))
unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_, y_output)))
R_squared = tf.subtract( 1.0, tf.div(total_error, unexplained_error))


rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, y_output))))
##
train_step = tf.train.AdamOptimizer(1e-4).minimize(rmse)

##train_step = tf.train.AdamOptimizer(1e-4).minimize(R_squared)
##correct_prediction = tf.equal(tf.argmax(y_output,0), tf.argmax(y_,0))
##correct_prediction2 = tf.subtract(y_, y_output)
##accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
##accuracy2 = tf.reduce_mean(correct_prediction2)




sess.run(tf.global_variables_initializer())

for i in range(200000):##25000

    Xtr, Ytr = next_batch(1000, X_train, y)
 
    Ytr = np.reshape(Ytr,(1000,1))
    if i%100 == 0:
      train_error_RMSE = sess.run(rmse, feed_dict={x:Xtr, y_:Ytr})
      train_error_R2 = sess.run(R_squared, feed_dict={x:Xtr, y_:Ytr, keep_prob_fc1: 1})
      
      print("step %d, training error RMSE: %g"%(i, train_error_RMSE))
      print("step %d, training error R2: %g"%(i, train_error_R2))
    train_step.run(feed_dict={x: Xtr, y_: Ytr})
##    print(y_output.eval())

##print("GENERATING PREDICTION FOR TEST FILE:")
##print(sess.run(y_output,feed_dict={x: X_test}))
##
##solution = pd.DataFrame({"ID":test.ID, "y": np.reshape(sess.run(y_output,feed_dict={x: X_test}),(4209,))})
##
##solution.to_csv("MercedesPredictionDeepNN.csv", index = False)


print("END OF PROGRAM")



##    model = Sequential()
##    # Input layer with dimension input_dims and hidden layer i with input_dims neurons. 
##    model.add(Dense(input_dims, input_dim=input_dims, activation='relu', kernel_constraint=maxnorm(3)))
##    model.add(BatchNormalization())
##    model.add(Dropout(0.2))
##    model.add(Activation("linear"))
##    # Hidden layer
##    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
##    model.add(BatchNormalization())
##    model.add(Dropout(0.3))
##    model.add(Activation("linear"))
##    # Hidden layer
##    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
##    model.add(BatchNormalization())
##    model.add(Dropout(0.3))
##    model.add(Activation("linear"))
##    # Hidden layer
##    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
##    model.add(BatchNormalization())
##    model.add(Dropout(0.3))
##    model.add(Activation("linear"))
##    # Hidden layer
##    model.add(Dense(input_dims//2, activation='relu', kernel_constraint=maxnorm(3)))
##    model.add(BatchNormalization())
##    model.add(Dropout(0.3))
##    model.add(Activation("linear"))
##    # Output Layer.
##    model.add(Dense(1))
