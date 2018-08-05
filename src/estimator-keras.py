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
imp = pp.Imputer(missing_values='NaN', strategy='median', axis=0) # probably, median will be better
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
X = combined_df.values

# Split Data Sets
test_n = df.shape[0]
x_train = X[:test_n,:]
y_train = y[:test_n]
x_test = X[test_n:,:]
y_test = y[test_n:]


### Start Keras
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model




















