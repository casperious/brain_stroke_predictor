# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:00:42 2023

@author: jerem
"""

"""
Working on Random Forest Regression Model as per https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9962068/#B14-jcdd-10-00082
"""

# Importing the libraries
import Preprocessor as preprocessor
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#Variables for code from preprocessor
fit = preprocessor.fit
dummified = preprocessor.dummified_set
stroke_col = preprocessor.stroke_col
cols = preprocessor.cols
d= preprocessor.d
d_o = preprocessor.d_o
x_train = preprocessor.x_train
y_train = preprocessor.y_train
x_one_train = preprocessor.x_one_train
y_one_train = preprocessor.y_one_train
x_test = preprocessor.x_test
x_one_test = preprocessor.x_one_test
y_test = preprocessor.y_test
y_one_test = preprocessor.y_one_test
st_x = preprocessor.st_x


classifier = RandomForestClassifier(n_estimators=20, random_state=1, max_depth=10)
classifier.fit(x_one_train,y_one_train)

y_one_pred = classifier.predict(x_one_test)

print(confusion_matrix(y_one_test,y_one_pred))
print(classification_report(y_one_test, y_one_pred))
print(accuracy_score(y_one_test, y_one_pred))
print("Mean Absolute Error = " , metrics.mean_absolute_error(y_one_test, y_one_pred))
print("Mean Squared Error = ", metrics.mean_squared_error(y_one_test, y_one_pred))
print("Root Mean Squared Error = ", np.sqrt(metrics.mean_squared_error(y_one_test, y_one_pred)))

#function to run classifier on new csv input
'''
Dictionary for One Hot Encoder in Preprocessor needs to be fit
'''
def ClassifyNew(data_set_new):
    # Using the dictionary to label future data
    #this should apply original encoding to new data
    for col in cols:
        print("Col is " , data_set_new[col])
        data_set_new[col] = d_o[col].transform(data_set_new[col])
    
    print(data_set_new.to_string())
    x_new = data_set_new.iloc[:,data_set_new.columns!=stroke_col].values
    x_scaled = st_x.fit_transform(x_new)
    prediction = classifier.predict(x_scaled)
    print("Prediction is No stroke") if prediction[0] == 0 else print("Prediction is Stroke")
