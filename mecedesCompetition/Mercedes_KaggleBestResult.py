

import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
##df.iloc[[2]]

##train_data_y_250 = train.ix[(train['y'] >= 250)]
##train_data_y_250_i = train.index[train['y'] >= 250]
##print(train_data_y_250)##[1 rows x 378 columns]
##temp = 0
##i=0


##print(train.iloc[[0]].drop(["ID", "y"], axis=1).columns.shape)##(376,)
##print(train_data_y_250.drop(["ID", "y"], axis=1).columns.shape)##(376,)

##print(train.iloc[[0]].drop(["ID", "y"], axis=1).reset_index(drop=True) == train_data_y_250.drop(["ID", "y"], axis=1).reset_index(drop=True))

##print(train.iloc[[0]].drop(["ID", "y"], axis=1).reset_index(drop=True).all())
##print(train_data_y_250.drop(["ID", "y"], axis=1).reset_index(drop=True).all())
##
##print(train.iloc[[0]].drop(["ID", "y"], axis=1).reset_index(drop=True)["X0"].all())
##print(train_data_y_250.drop(["ID", "y"], axis=1).reset_index(drop=True)["X0"].all())

##while i < len(train.index):
##    allSame = True
##    for value in train.drop(["ID", "y"], axis=1).columns:
##        
##        temp1 = train.iloc[[i]].drop(["ID", "y"], axis=1).reset_index(drop=True)[value].all()
##        temp2 = train_data_y_250.drop(["ID", "y"], axis=1).reset_index(drop=True)[value].all()
##        if temp1 != temp2:
##            allSame = False
##            break
##    if allSame == True:
##        print("ALLSAME AS THE OUTLIER ROW, INDEX: ", i)
####    else:
####        print("NOT SAME: ", i)
####    if train.iloc[[0]].drop(["ID", "y"], axis=1).reset_index(drop=True)["X0"] == train_data_y_250.drop(["ID", "y"], axis=1).reset_index(drop=True)["X0"]:
####        print("haha")
##    i = i + 1
##FOR TRAIN DATA, ONLY ONE ROW HAS THE SAME BINARY VALUE PATTERN AS 883: 883 itself

##print( test.iloc[[5]].drop(["ID"], axis=1).loc[:,'X10':'X385'])
##print(train_data_y_250.drop(["ID", "y"], axis=1).loc['X10':'X385']['X10'].any())
 
##i=0
##while i < len(test.index):
##    allSame = True
##    for value in test.drop(["ID"], axis=1).loc[:,'X10':'X385'].columns:
##        
##        temp1 = test.iloc[[i]].drop(["ID"], axis=1).reset_index(drop=True).loc[:,'X10':'X385'][value].all()
##        temp2 = train_data_y_250.drop(["ID", "y"], axis=1).reset_index(drop=True).loc[:,'X10':'X385'][value].all()
##        if temp1 != temp2:
##            allSame = False
##            break
##    if allSame == True:
##        print("ALLSAME AS THE OUTLIER ROW, INDEX: ", i)
##    else:
##        print("NOT SAME: ", i)
####    if train.iloc[[0]].drop(["ID", "y"], axis=1).reset_index(drop=True)["X0"] == train_data_y_250.drop(["ID", "y"], axis=1).reset_index(drop=True)["X0"]:
####        print("haha")
##    i = i + 1    
    









for c in train.columns:
    if train[c].dtype == 'object':
##        print(train[c])
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))



n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
print(tsvd_results_train.shape)##(4209, 12)
tsvd_results_test = tsvd.transform(test)
print(tsvd_results_test.shape)##(4209, 12)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

#save columns list before adding the decomposition components

##print(type(set(train.columns)))##<class 'set'>
##print(type(set(['y'])))##<class 'set'>
usable_columns = list(set(train.columns) - set(['y']))
##print(type(usable_columns))##<class 'list'>


##print(pca2_results_train[:, 2].shape)##(4209,)
# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]



##train = train[train['y']<250]    
y_train = train['y'].values


y_mean = np.mean(y_train)
id_test = test['ID'].values
#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].drop("ID", axis=1).values
finaltestset = test[usable_columns].drop("ID", axis=1).values


##'''Train the xgb model then predict the test data'''
lowestError = 10000000000
bestRound = 0


xgb_params = {
##    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}
##6.93391
xgb_params = {
##    'n_trees': 520, 
    'eta': 0.001,
    "gamma": 5.0,
    'min_child_weight':100,
    'max_depth': 7,
    'subsample': 0.9,
    "colsample_bytree":0.9,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}
earlyStoppingRounds = 50

X_train, X_test, y_train, y_test = train_test_split(train, train['y'].values, test_size=0.20, random_state=7)

### NOTE: Make sure that the class is labeled 'class' in the data file
##
##dtrain = xgb.DMatrix(train.drop(['y','ID'], axis=1), y_train)
dtrain = xgb.DMatrix(X_train.drop(['y','ID'], axis=1), y_train)

##dtest = xgb.DMatrix(test)
dtest = xgb.DMatrix(X_test.drop(['y','ID'], axis=1), y_test)

##

watchlist = [ (dtrain,'train'), (dtest, 'test') ]
##watchlist = [ (dtrain,'train') ]

num_boost_rounds = 1250
### train model
print("TRAINING XGBOOST MODEL......")


model = xgb.cv(xgb_params, xgb.DMatrix(train.drop(["y","ID"], axis=1), train['y'].values),  num_boost_round=10000000000,
        early_stopping_rounds= earlyStoppingRounds, nfold = 5, verbose_eval=5, seed=10)

print("ERROR: ", model.values[model.shape[0]-1][0])
print("BEST ROUND: ", model.shape[0])

##model = xgb.train(dict(xgb_params, silent=0), dtrain,  10000000000, watchlist,  early_stopping_rounds=earlyStoppingRounds, verbose_eval=5)
##y_pred = model.predict(dtest)


####
####'''Train the stacked models then predict the test data'''
####
##stacked_pipeline = make_pipeline(
##    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
##    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
##    LassoLarsCV()
##
##)
##
##print("TRAINING STACKED MODEL......")
##stacked_pipeline.fit(finaltrainset, y_train)
##results = stacked_pipeline.predict(finaltestset)
##
####'''R2 Score on the entire Train data when averaging'''
##
##print('R2 score on train data:')
##print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))
##
####'''Average the preditionon test data  of both models then save it on a csv file'''
##
##sub = pd.DataFrame()
##sub['ID'] = id_test
##sub['y'] = y_pred*0.75 + results*0.25
##sub.to_csv('mecedesPredict.csv', index=False)

print("END OF PROGRAM")

##ID = 1: 71.34112
##ID = 12: 109.30903
##ID = 23: 115.21953
##ID = 28: 92.00675
##ID = 42: 87.73572
##ID = 43: 129.79876
##ID = 45: 99.55671
##ID = 57: 116.02167
##ID = 88: 90.33211
##ID = 89: 130.55165
##ID = 93: 105.79792
##ID = 94: 103.04672
##ID = 104: 92.37968
##ID = 437: 85.9696
##ID = 1001: 111.65212
##ID = 3977: 132.08556
