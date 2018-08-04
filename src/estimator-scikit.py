#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as metrics

# df == data frame - this is the data that we are trying to predict
df = pd.read_csv('../input/train.csv')
df = df.set_index('Id')
print(df.head())

# sdf == test data frame - this has the actual data to use to make predictions
sdf = pd.read_csv('../input/test.csv')
sdf = sdf.set_index('Id')
print(sdf.head())

# df.columns.values is the values of the csv
print(df.columns.values)

# get the SalePrice Column from the dataframe and pretty print it
y = df.SalePrice
print("Average sale price: " + "${:,.0f}".format(y.mean()))

# -- combine test and training data for pre-processing

# get only the sales prices from the df dataset
df = df.drop('SalePrice', axis=1)

# append will add another dataframe to this dataframe
all_df = df.append(sdf)

# list (#of rows, #of columns)
print(all_df.shape)

# -- end combine test and trianing data for pre-processing

all_features = list(df.columns.values)
numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
'BsmtFinSF2', 'BsmtUnfSF','LowQualFinSF','GrLivArea','BsmtFullBath',
'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
'TotRmsAbvGrd','TotalBsmtSF','Fireplaces', 'GarageCars', 'GarageArea',
'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
'PoolArea', 'MiscVal']

# <map syntax> for-each <f> in <collection>
# if <filter syntax( -> not(f exists in numeric_features))>
categorical_features = [f for f in all_features if not(f in numeric_features)]

# print the length of our three list of features
print(len(all_features), len(categorical_features), len(numeric_features))

# create a dataframe that's composed only of the numeric features of the dataset
numeric_df = all_df[numeric_features]
print(numeric_df.shape)

# imputing the data to infer a value (in this case the most frequent number will be imputed)
X = numeric_df.values

# create the definition of the imputer
imp = pp.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# We want to 'fit' the imputer usually on a smaller dataset - then apply a transform on the
# larget dataset - but in this case the dataset is sufficiently small
imp = imp.fit(X)

# apply the inputer by transforming our whole dataset
X = imp.transform(X)

print(X.shape)

# scaling
scaler = pp.StandardScaler()
scaler = scaler.fit(X)
X = scaler.transform(X)

# print row 0, with (: == all) columns
print(X[0, :])

# Expand categorical data into comlumns -
def process_categorical(ndf, df, categorical_features):
    for f in categorical_features:
        # a dummy is a value (1 or 0) if the property is present, similar to an enumeration
        new_cols = pd.DataFrame(pd.get_dummies(df[f]))
        new_cols.index = df.index
        ndf = pd.merge(ndf, new_cols, how = 'inner', left_index=True, right_index=True)
    return ndf

numeric_df = pd.DataFrame(X)

numeric_df.index = all_df.index
combined_df = process_categorical(numeric_df, all_df, categorical_features)
print(df['SaleCondition'].head())
print(set(df["SaleCondition"].values))
print(combined_df[['AdjLand', 'Family', 'Partial', 'Abnorml', 'Normal', 'Alloca']].head())

print(numeric_df.columns.values)
print(X[0, :])

X = combined_df.values
print(X.shape)

#PCA
from sklearn.decomposition import PCA

test_n = df.shape[0]
x = X[:test_n,:]

print("applying pca fitting")
pca = PCA()
pca_fitted = pca.fit(x)
x = pca_fitted.transform(x)

x_test = X[test_n:,:]
y_test = y[test_n:]

x_train = X[:test_n,:]
y_train = y[:test_n]

from sklearn import linear_model

print("fitting linear regression")
lr = linear_model.LinearRegression()

lr.fit(x_train, y_train)

print("fitting ridge regression")
ridge = linear_model.Ridge()
ridge.fit(x_train, y_train)


x_value = x_test
y_value = lr.predict(x_test)

y_value_ridge = ridge.predict(x_test)

print(x_test.shape)

# The test data does not have a value to test against?
print('Linear Regression score is %f' % lr.score(x_value, y_value))
print('Linear Regression relative to ridge is %f' % lr.score(x_value, y_value_ridge))
print('Ridge score is %f' % ridge.score(x_value, y_value))
print('Ridge score relative to itself is %f' % ridge.score(x_value, y_value_ridge))
