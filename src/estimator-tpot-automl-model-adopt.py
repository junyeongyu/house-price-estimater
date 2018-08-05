#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 0. Import
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as metrics

# Load Data Sets
df = pd.read_csv('../input/train.csv')
df = df.set_index('Id')
y = df.SalePrice

all_df = df.drop('SalePrice', axis=1)

df = all_df.iloc[:1000, :]
sdf = all_df.iloc[1000:, :]


# Create lists of categorical vs numeric features
all_features = list(df.columns.values)
numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','TotalBsmtSF','Fireplaces', 'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
#numeric_features = list(df.select_dtypes(include=[np.number]).columns.values)
categorical_features = [f for f in all_features if not (f in numeric_features)]
numeric_df = all_df[numeric_features]

# Impute
X = numeric_df.values
imp = pp.Imputer(missing_values='NaN', strategy='most_frequent', axis=0) # probably, median will be better
imp = imp.fit(X)
X = imp.transform(X)

# Expand categorical into columns
def process_categorical(ndf, df, categorical_features):
    for f in categorical_features:
        new_cols = pd.DataFrame(pd.get_dummies(df[f]))
        new_cols.index = df.index
        ndf = pd.merge(ndf, new_cols, how = 'inner', left_index=True, right_index=True)
    return ndf

numeric_df = pd.DataFrame(X)
numeric_df.index = all_df.index
combined_df = process_categorical(numeric_df, all_df, categorical_features)

X = combined_df.values

# Split Data Sets
#x_train = df.values
#y_train = y[:1000]
#x_test = sdf.values
#y_test = y[1000:]

test_n = df.shape[0]
x_train = X[:test_n,:]
y_train = y[:test_n]
x_test = X[test_n:,:]
y_test = y[test_n:]


#import numpy as np
#import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
#tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
#features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = x_train, x_test, y_train, y_test #\
#            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-782999502.5452403
model = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    RandomForestRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=5, min_samples_split=12, n_estimators=100)
)

model.fit(training_features, training_target)
results = model.predict(testing_features)
#print(results)


y_pred = model.predict(x_test[:10,])
print (y_pred)
print (y_test[:10])
print('MLPRegressor Regression score is %f (traning)' % model.score(x_train, y_train))
print('MLPRegressor Regression score is %f (test)' % model.score(x_test, y_test)) # 84%


