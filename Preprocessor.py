# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:03:51 2023

@author: jerem
"""

"""
Data Preprocessor

"""
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import category_encoders as ce
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
gender_categories = ['Male', 'Female']
ever_married_categories= ['Yes','No']
work_type_categories = ['Private', 'Self Employed', 'Govt_job', 'children']
Residence_type_categories = ['Rural', 'Urban']
Smoking_status_categories = ['Unknown', 'formerly smoked', 'never smoked', 'smokes']
col_indices = [0,4,5,6,9]

#encoding categorical data
d = defaultdict(LabelEncoder)
d_o = defaultdict(OneHotEncoder)
ohe = OneHotEncoder(drop = 'first')
# Encoding the variable

fit = data_set.apply(lambda x: d[x.name].fit_transform(x))
fit.apply(lambda x: d[x.name].inverse_transform(x))
#data_set.apply(lambda x: d_o[x.name].fit_transform(x))
#print(d.keys())
#print("________________")
#print(d_o.keys())
'''
One Hot Encoding with dummy vars

'''

'''def dummify(OHE, x, columns):
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
dummified = model_OHE.fit(data_set)'''


# Create a categorical boolean mask
categorical_feature_mask = data_set.dtypes == object
# Filter out the categorical columns into a list for easy reference later on in case you have more than a couple categorical columns
categorical_cols = data_set.columns[categorical_feature_mask].tolist()

# Instantiate the OneHotEncoder Object
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse = False)
# Apply ohe on data
ohe.fit(data_set[categorical_cols])
cat_ohe = ohe.transform(data_set[categorical_cols])

#Create a Pandas DataFrame of the hot encoded column
ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = categorical_cols))
#concat with original data and drop original columns
df_ohe = pd.concat([data_set, ohe_df], axis=1).drop(columns = categorical_cols, axis=1)

dummified_set = df_ohe
#dummified_one_hot = ohe.fit_transform(data_set)
#dummified_set = pd.get_dummies(data = data_set, columns = cols)
#dummies_frame_one = pd.get_dummies(data = data_set, columns = cols)
#print(dummified.head())
#dummified_set = dummify(model_OHE, data_set,cols)
#fit_set = dummified_set.apply(lambda x: d_o[x.name].fit_transform(x))
#fit_set.apply(lambda x: d_o[x.name].inverse_transform(x))

'''
Below is independent and dependent split of label encoded data_set
'''
#extracting independent and dependent Variable
x = fit.iloc[:,fit.columns!=stroke_col].values  #all columns excluding dependent variable
y = fit.iloc[:,10].values     #dependent variable

'''
Below is independent and dependent split of one hot encoded dummy var data set
'''
x_one = dummified_set.iloc[:,dummified_set.columns!=stroke_col].values
y_one = dummified_set.iloc[:,5].values


#splitting dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 4)
x_one_train, x_one_test, y_one_train, y_one_test = train_test_split(x_one,y_one, test_size = 0.25, random_state = 2)

#feature scaling
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)
x_one_train = st_x.fit_transform(x_one_train)
x_one_test = st_x.fit_transform(x_one_test)
