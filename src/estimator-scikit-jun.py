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

#sdf = pd.read_csv('../input/test.csv')
#sdf = sdf.set_index('Id')
#print(df.head())

#print(df.Street)
#print (df.columns.values)

y = df.SalePrice
print("Average sale price: " + "${:,.0f}".format(y.mean()))

# Combine test and train for preprocessing
#df = df.drop('SalePrice', axis=1)
#all_df = df.append(sdf)
all_df = df.drop('SalePrice', axis=1)
df = all_df.iloc[:1000, :]
sdf = all_df.iloc[1000:, :]
print(all_df.shape)
print(df.shape)
print(sdf.shape)


# Create lists of categorical vs numeric features
all_features = list(df.columns.values)
numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','TotalBsmtSF','Fireplaces', 'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
#numeric_features = list(df.select_dtypes(include=[np.number]).columns.values)
categorical_features = [f for f in all_features if not (f in numeric_features)]
print(len(all_features), len(categorical_features), len(numeric_features))

# Preprocess numerical columns
numeric_df = all_df[numeric_features]
print(numeric_df.shape)

# Impute
X = numeric_df.values
imp = pp.Imputer(missing_values='NaN', strategy='most_frequent', axis=0) # probably, median will be better
imp = imp.fit(X)
X = imp.transform(X)
print(X.shape)

# Scale
#print (X[0, :])
scaler = pp.StandardScaler() # StandardScaler, MinMaxScaler
scaler = scaler.fit(X)
X = scaler.transform(X)
#print (X[0, :])

# PCA
from sklearn.decomposition import PCA
pca = PCA()
X = pca.fit_transform(X)

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

print(df['SaleCondition'].head())
print (set(df["SaleCondition"].values))
print(combined_df[['AdjLand', 'Family', 'Partial', 'Abnorml', 'Normal', 'Alloca']].head())

X = combined_df.values
print(X.shape)

# Split Data Sets
test_n = df.shape[0]
x_train = X[:test_n,:]
y_train = y[:test_n]
x_test = X[test_n:,:]
y_test = y[test_n:]

# 1-1. Train data (with Logistic Regression)
from sklearn import linear_model

''''''
print ('training...')
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test[:10,])
print (y_pred)
print (y_test[:10])
print('LinearRegression Regression score is %f (traning)' % model.score(x_train, y_train))
print('LinearRegression Regression score is %f (test)' % model.score(x_test, y_test)) # 65%

# 1-2. Train data (with nural network)
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(
    hidden_layer_sizes=(200, 100, 30, ), activation='relu')
print ('training...')
model.fit(x_train, y_train)
y_pred = model.predict(x_test[:10,])
print (y_pred)
print (y_test[:10])
print('MLPRegressor Regression score is %f (traning)' % model.score(x_train, y_train))
print('MLPRegressor Regression score is %f (test)' % model.score(x_test, y_test)) # 75%

