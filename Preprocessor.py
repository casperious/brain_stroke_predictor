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
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.utils import resample
data = pd.read_csv('brain_stroke.csv')
#data_set = data
data_set = shuffle(data)
#data_set.fillna("Unknown")
stroke_col = 'stroke'
x_cols = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
y_cols = ['stroke']
cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
gender_categories = ['Male', 'Female']
ever_married_categories= ['Yes','No']
work_type_categories = ['Private', 'Self Employed', 'Govt_job', 'children']
Residence_type_categories = ['Rural', 'Urban']
Smoking_status_categories = ['Unknown', 'formerly smoked', 'never smoked', 'smokes']
col_indices = [0,4,5,6,9]

#print(data_set['stroke'].value_counts())

'''
Upsampling positive stroke data to match negative stroke
'''
df_majority = data_set[(data_set['stroke']==0)]
df_minority = data_set[(data_set['stroke']==1)]

#upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace = True,        #sample with replacement
                                 n_samples = 4733,      #To match majority count
                                 random_state = 42)     #reproducible results

#combine majority with upsampled minority data
df_upsampled = pd.concat([df_minority_upsampled, df_majority])
#print(df_upsampled['stroke'].value_counts())
'''
Below is independent and dependent split of label encoded data_set
'''
#extracting independent and dependent Variable
x = df_upsampled.iloc[:,data_set.columns!=stroke_col].values  #all columns excluding dependent variable
y = df_upsampled.iloc[:,10].values     #dependent variable


x = pd.DataFrame(x, columns = x_cols)
#splitting dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 4)

#encoding categorical data
ohe = OneHotEncoder(drop = 'first')

# Create a categorical boolean mask
categorical_feature_mask = data_set.dtypes == object
# Filter out the categorical columns into a list for easy reference later on in case you have more than a couple categorical columns
categorical_cols = data_set.columns[categorical_feature_mask].tolist()

# Instantiate the OneHotEncoder Object
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output = False)
# Apply ohe on data
ohe.fit(x_train[categorical_cols])
cat_ohe = ohe.transform(x_train[categorical_cols])

#Create a Pandas DataFrame of the hot encoded column
ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = categorical_cols))
# reset indices to prevent NaN's and concat with original data and drop original columns 
ohe_df.reset_index(drop=True,inplace=True)
x_train.reset_index(drop=True,inplace=True)
x_one_train_df = pd.concat([x_train,ohe_df],axis = 1).drop(columns=categorical_cols,axis=1)

x_one_test = ohe.transform(x_test[categorical_cols])
#Create a Pandas DataFrame of the hot encoded column
ohe_df_test = pd.DataFrame(x_one_test, columns = ohe.get_feature_names_out(input_features = categorical_cols))
#concat with original data and drop original columns
x_test.reset_index(drop=True, inplace=True)
ohe_df_test.reset_index(drop=True, inplace=True)
x_one_test = pd.concat([x_test, ohe_df_test], axis=1).drop(columns = categorical_cols, axis=1)

#feature scaling
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
st_x.fit(x_one_train_df)
x_one_train = st_x.transform(x_one_train_df)
x_one_test = st_x.transform(x_one_test)
