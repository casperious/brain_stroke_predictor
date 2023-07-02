# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:03:51 2023

@author: jerem
"""

"""
Data Preprocessor

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import category_encoders as ce
import sklearn as skl
from sklearn import preprocessing
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer


data_set = pd.read_csv('brain_stroke.csv')

stroke_col = 'stroke'
cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
col_indices = [0,4,5,6,9]

#encoding categorical data
d = defaultdict(LabelEncoder)
d_o = defaultdict(OneHotEncoder)
ohe = OneHotEncoder(categories = col_indices, drop = 'first', sparse = False)
# Encoding the variable
fit = data_set.apply(lambda x: d[x.name].fit_transform(x))
fit.apply(lambda x: d[x.name].inverse_transform(x))


'''
One Hot Encoding with dummy vars

'''

def dummify(OHE, x, columns):
    transformed_array = OHE.transform(x)
    initial_colnames_keep = list(set(x.columns.tolist()) - set(columns))
    new_colnames = np.concatenate(model_OHE.named_transformers_['OHE'].categories_).tolist()
    all_colnames = new_colnames + initial_colnames_keep 
    df = pd.DataFrame(transformed_array, index = x.index, columns = all_colnames)
    return df

model_OHE = ColumnTransformer(
    [('OHE', OneHotEncoder(),['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])],
    remainder = 'passthrough'
    )
dummified = model_OHE.fit(data_set)
dummified = dummify(model_OHE, data_set,cols)