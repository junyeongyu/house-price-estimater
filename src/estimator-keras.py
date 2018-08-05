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
#print(X.shape)

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


arr_x_train = x_train
arr_y_train = y_train
#kc_y_valid = y_test
#kc_x_valid = x_test

#y_train = np.array(y_train)

#print('Size of training set: ', len(kc_x_train))
#print('Size of validation set: ', len(kc_x_valid))
#print('Size of test set: ', len(kc_test), '(not converted)')

def norm_stats(df1, df2):
    dfs = df1.append(df2)
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return (minimum, maximum, mu, sigma)

def z_score(col, stats):
    m, M, mu, s = stats
    df = pd.DataFrame()
    for c in col.columns:
        df[c] = (col[c]-mu[c])/s[c]
    return df


def basic_model_x(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(y_size))
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)

def basic_model_1(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="relu", input_dim=x_size))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(y_size))
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)

def basic_model_2(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="relu", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(20, activation="relu"))
    t_model.add(Dense(y_size))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)

def basic_model_3(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(80, activation="relu", kernel_initializer='normal', input_shape=(x_size,)))
    t_model.add(Dropout(0.3))
    t_model.add(Dense(120, activation="relu", kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0.3))
    t_model.add(Dense(20, activation="relu", kernel_initializer='normal', kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0.3))
    t_model.add(Dense(10, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(y_size))
    t_model.compile(
        loss='mean_squared_error',
        optimizer='nadam',
        metrics=[metrics.mae])
    return(t_model)

print(x_train.shape[1])
print(y_train.shape)

# Make mode
#model = basic_model_2(x_train.shape[1], 1)
#model.summary()

def make_model():
    return basic_model_3(x_train.shape[1], 1)


epochs = 1000 #500
batch_size = 1000
print('Epochs: ', epochs)
print('Batch size: ', batch_size)

keras_callbacks = [
    # ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=2)
    # ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0)
    # TensorBoard(log_dir='/tmp/keras_logs/model_3', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),
    EarlyStopping(monitor='val_mean_absolute_error', patience=80, verbose=0) # 20
]
print(x_train.shape)

#keras.wrappers.scikit_learn.KerasRegressor

from keras.wrappers.scikit_learn import KerasRegressor
model = KerasRegressor(build_fn=make_model, epochs=epochs, batch_size=batch_size, verbose=True, callbacks=keras_callbacks)
model.fit(x_train, y_train)

'''
history = model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2,#0, # Change it to 2, if wished to observe execution
    #validation_data=(arr_x_valid, arr_y_valid),
    callbacks=keras_callbacks)
'''
y_pred = model.predict(x_test[:20,])

print (y_pred)
print (y_test[:20])
print('KERAS score is %f (traning)' % model.score(x_train, y_train))
print('KERAS score is %f (test)' % model.score(x_test, y_test)) # ??%

'''
from sklearn.metrics import mean_squared_error
score = mean_squared_error(y_test, model.predict(x_test))
print(score)
from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(y_test, model.predict(x_test))
print(score)
'''

