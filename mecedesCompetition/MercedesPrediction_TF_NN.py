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
sess = tf.InteractiveSession()

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

all_data = pd.concat((train.loc[:,'X0':'X385'],
                      test.loc[:,'X0':'X385']))


all_data = pd.get_dummies(all_data)
##print(all_data.describe())
##print(all_data.head())
##print(train.shape[0])##4209
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.y

testIDs = test.ID
##print(testIDs)
##print(type(testIDs))
##print(testIDs.values)
##print(np.concatenate(testIDs.values, testIDs.values))
print(np.vstack((testIDs.values,testIDs.values)).T)

X_train = X_train.values
##print(X_train.shape)##(4209, 579)
X_test = X_test.values
##print(X_test.shape)##(4209, 579)
y = y.values
##print(y.shape)##(4209,)
##print(type(X_train))##<class 'pandas.core.frame.DataFrame'>
##print(type(X_test))##<class 'pandas.core.frame.DataFrame'>
##print(type(y))##<class 'pandas.core.series.Series'>





def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
##    float_data_shuffle = np.vstack(np.asarray(data_shuffle)).astype(np.float)
##    float_labels_shuffle = np.vstack(np.asarray(labels_shuffle)).astype(np.float)
##    return np.asarray(float_data_shuffle), np.asarray(float_labels_shuffle)
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

##Xtr, Ytr = np.arange(0, 10), np.arange(0, 100).reshape(10, 10)
##print(Xtr)
##print(Ytr)
##
##Xtr, Ytr = next_batch(5, Xtr, Ytr)
##print(type(Xtr))##<class 'numpy.ndarray'>
##print(type(Ytr))##<class 'numpy.ndarray'>
##print('\n5 random samples')
##print(Xtr)
##print(Ytr)

##Xtr, Ytr = next_batch(5, X_train, y)
##print('\n5 random samples')
##print(Xtr)
##print(Xtr.dtype)
##print(Xtr.shape)
##
##print(Ytr)
##print(Ytr.dtype)
##print(Ytr.shape)






##BELOW IS THE TENSOR FLOW PART OF THIS PROGRAM
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 579])##holder for training data
y_ = tf.placeholder(tf.float32, shape=[None, 1])##holder for truth values

W_fc1 = weight_variable([579,1])
b_fc1 = bias_variable([1])

##x_initialInput = tf.reshape(x,[-1,579])

y_output = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
##y_output = tf.matmul(x, W_fc1)
##print(y_output.get_shape())##(?, 1)


rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, y_output))))

train_step = tf.train.AdamOptimizer(1e-4).minimize(rmse)

correct_prediction = tf.equal(tf.argmax(y_output,0), tf.argmax(y_,0))
##
correct_prediction2 = tf.subtract(y_, y_output)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy2 = tf.reduce_mean(correct_prediction2)


prediction=tf.argmax(y_output,0)


sess.run(tf.global_variables_initializer())

for i in range(1):##25000
##    print("Epoch: ",i)
######  batch = mnist.train.next_batch(100)
######  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    Xtr, Ytr = next_batch(100, X_train, y)
##    
    Ytr = np.reshape(Ytr,(100,1))
####    print(Ytr)
####    print(Xtr.shape)##(100, 579)
####    print(Xtr.dtype)##int64
####    print(Ytr.shape)##(100,)
####    print(Ytr.dtype)##object
####    Xtr = Xtr.astype(np.float32)
####    Ytr = Ytr.astype(np.float32)
####    Xtr = np.asfarray(Xtr)
####    print(Xtr.shape)##(100, 579)
####    print(Xtr.dtype)##int64
####    Ytr = np.asfarray(Ytr)
####    print(Ytr.shape)##(100,)
####    print(Ytr.dtype)##object
##    
    if i%100 == 0:
      train_error = sess.run(rmse, feed_dict={x:Xtr, y_:Ytr})
      print("step %d, training error %g"%(i, train_error))
##      train_accuracy = accuracy2.eval(feed_dict={
##        x:Xtr, y_:Ytr})
##      print("step %d, training error %g"%(i, train_accuracy))
##      print(sess.run(y_output,feed_dict={x: X_test}))
##      print (prediction.eval(feed_dict={x: X_test}, session=sess))
####    print(tf.argmax(y,1))
####    print(tf.argmax(y_,1))
####    y_output.run(feed_dict={x: Xtr, y_: Ytr})
####    print(type(y_output))
##    print(y_output.get_shape())
    train_step.run(feed_dict={x: Xtr, y_: Ytr})
##    print(y_output.eval())
print("AFTER TRAINING, THE AVERAGE ERROR: ")    
print(accuracy2.eval(feed_dict={x: X_train, y_: np.reshape(y,(4209,1))}))

print("GENERATING PREDICTION FOR TEST FILE:")
print(sess.run(y_output,feed_dict={x: X_test}))
##print(sess.run(tf.argmax(y_output,1),feed_dict={x: X_test}) )
##prediction.eval(feed_dict={x: mnist.test.images}, session=sess)
##print (prediction.eval(feed_dict={x: X_test}, session=sess))
##print (type(prediction.eval(feed_dict={x: X_test}, session=sess)))##<class 'numpy.ndarray'>
##print (prediction.eval(feed_dict={x: X_test}, session=sess).shape)##(4209,)
##Ytr = np.reshape(Ytr,(100,1))
##print(type(sess.run(y_output,feed_dict={x: X_test})))##<class 'numpy.ndarray'>
##print(sess.run(y_output,feed_dict={x: X_test}).shape)
##print(np.reshape(sess.run(y_output,feed_dict={x: X_test}),(4209,)))

##np.concatenate((sess.run(y_output,feed_dict={x: X_test}), testIDs), axis=1)
##print(np.concatenate(testIDs, testIDs))
##print(np.vstack((testIDs.values, np.reshape(sess.run(y_output,feed_dict={x: X_test}),(4209,))       )).T)

##X = np.vstack((testIDs.values, np.reshape(sess.run(y_output,feed_dict={x: X_test}),(4209,))       )).T

##np.savetxt("MercedesPredictionSimpleNN.csv", X)

solution = pd.DataFrame({"ID":test.ID, "y": np.reshape(sess.run(y_output,feed_dict={x: X_test}),(4209,))})

solution.to_csv("MercedesPredictionNN.csv", index = False)


print("END OF PROGRAM")
