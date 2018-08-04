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

sdf = pd.read_csv('../input/test.csv')
sdf = sdf.set_index('Id')
df.head(5)

#print(df.Street)
print (df.columns.values)

y = df.SalePrice
print("Average sale price: " + "${:,.0f}".format(y.mean()))