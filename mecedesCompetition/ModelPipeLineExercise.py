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
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import r2_score
from sklearn.svm import SVR


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


for c in train.columns:
    if train[c].dtype == 'object':
##        print(train[c])
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

##print(train.values.shape)
##print(train['X0'])

#save columns list before adding the decomposition components
usable_columns = list(set(train.columns) - set(['y','ID']))
##print(usable_columns)        

##'''''''''''''''''''''''''''''''''''''''''''''''Train the stacked models then predict the test data'''''''''''''''''''''''''''''''''''''''''''''''

#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
##print(finaltrainset.shape)##(4209, 377)
finaltestset = test[usable_columns].values
##print(finaltestset.shape)##(4209, 377)
y_train = train['y'].values

##logistic = linear_model.LogisticRegression()
##LassoLarsCV = LassoLarsCV()

##TSVD = decomposition.TruncatedSVD(n_components = 300)
##PCA = decomposition.PCA(n_components = 8)
##ICA = FastICA(n_components=12, random_state=420)
##GRP = GaussianRandomProjection(n_components=12, eps=0.1, random_state=420)
##SRP = SparseRandomProjection(n_components=12, dense_output=True, random_state=420)
##anova_filter = SelectKBest(f_regression, k=3)
##pipe = Pipeline(steps=[('pca', PCA),
##                       ('LassoLarsCV', LassoLarsCV)])


###############################################################################
### Plot the PCA spectrum
##firstColumns = ['X0','X1','X2','X3','X4','X5','X6','X8']
##
##TSVD.fit(train[usable_columns].drop(firstColumns, axis=1), train['y'].values)
##PCA.fit(train[usable_columns][firstColumns], train['y'].values)
##ICA.fit(finaltrainset, train['y'].values)
##GRP.fit(finaltrainset, train['y'].values)
##SRP.fit(finaltrainset, train['y'].values)
##
##anova_filter.fit(train[usable_columns][firstColumns].values, train['y'].values)
####print(list(zip(anova_filter.get_support(), usable_columns)))##(377,)
##for value in list(zip(anova_filter.get_support(), firstColumns)):
##    if value[0] == True:
##        print(value[1])
##    
##plt.figure(1, figsize=(4, 3))
##plt.clf()
##plt.axes([.2, .2, .7, .7])
####plt.plot(PCA.explained_variance_ratio_, linewidth=2)
##plt.plot(TSVD.explained_variance_ratio_, linewidth=2)
##plt.axis('tight')
##plt.xlabel('n_components')
##plt.ylabel('explained_variance_')
##plt.show()



























































###############################################################################
# Prediction
##LassoLarsCV = LassoLarsCV()
##PCA = decomposition.PCA()
##TSVD = decomposition.TruncatedSVD()
##pipe = Pipeline(steps=[('pca', PCA),
##                       ('LassoLarsCV', LassoLarsCV)])
##pipe = Pipeline(steps=[('TSVD', TSVD),
##                       ('LassoLarsCV', LassoLarsCV)])
##n_components_PCA = [20, 40, 64]
##n_components_TSVD = [5, 10, 15]
##
##Cs = np.logspace(-4, 4, 3)
##
##estimator = GridSearchCV(pipe,
##                         dict(
####                             pca__n_components=n_components_PCA,
##                             TSVD__n_components=n_components_TSVD,
##                              ))
##
##estimator.fit(finaltrainset, train['y'].values)
##
##plt.axvline(estimator.best_estimator_.named_steps['TSVD'].n_components,
##            linestyle=':', label='n_components_chosen')
##plt.legend(prop=dict(size=20))
##plt.show()


##<------------------------------------------------BEST KAGGLE RESULT PIPE-------------------------------------------------



##print(finaltrainset.shape)
featureExtraction_estimators = [("tsvd", decomposition.TruncatedSVD()),
                                ('pca', decomposition.PCA()),
                                ('grp', GaussianRandomProjection()),
                                ('ica', FastICA()),
                                ('srp', SparseRandomProjection()),
                                ('knn', SelectKBest(f_regression)),                                
                                ]
combinedFeatureExtractors = FeatureUnion(featureExtraction_estimators)

##print(combinedFeatureExtractors.get_params().keys())
##X_features = combinedFeatureExtractors.fit(finaltrainset, y_train).transform(finaltrainset)
##print(X_features.shape)
pipe = Pipeline([("extractfeatures", combinedFeatureExtractors),
                 ("GBR", GradientBoostingRegressor())##LassoLarsCV(), GradientBoostingRegressor()
                ])

pipe = Pipeline([("extractfeatures", combinedFeatureExtractors),
                 ("finaloutput", GradientBoostingRegressor())##LassoLarsCV(), GradientBoostingRegressor()
                ])

##pipe.fit(finaltrainset, y_train)

param_grid = dict(extractfeatures__tsvd__n_components=[1, 2, 3],
                  extractfeatures__pca__n_components=[1, 2, 3],
                  extractfeatures__grp__n_components=[1, 2, 3],
                  extractfeatures__ica__n_components=[1, 2, 3],
                  extractfeatures__srp__n_components=[1, 2, 3],
                  extractfeatures__knn__k=[1, 2, 3],
                  finaloutput=[GradientBoostingRegressor(), SVR()],
##                  finaloutput__max_depth = [3,4,5],
                  
                  )

grid_search = GridSearchCV(pipe, param_grid=param_grid, verbose=10)
##print(grid_search.get_params().keys())
grid_search.fit(finaltrainset, y_train)
print(grid_search.best_estimator_)

##'''''''''''''''''''''''''''''''''''''''''''''''R2 Score on the entire Train data when averaging''''''''''''''''''''''''''''''''''''''''''''''''''

print('R2 score on train data:')
print(r2_score(y_train, grid_search.predict(finaltrainset)))






print("END OF PROGRAM")
