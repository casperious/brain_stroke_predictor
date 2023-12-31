# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:00:42 2023

@author: jerem
"""

"""
Working on Random Forest Regression Model as per https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9962068/#B14-jcdd-10-00082


Weighted Random Forest https://arxiv.org/ftp/arxiv/papers/2009/2009.00534.pdf
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
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

data_set = preprocessor.data_set
stroke_col = preprocessor.stroke_col
cols = preprocessor.cols
x_train = preprocessor.x_train
y_train = preprocessor.y_train
x_one_train = preprocessor.x_one_train
x_test = preprocessor.x_test
x_one_test = preprocessor.x_one_test
y_test = preprocessor.y_test
st_x = preprocessor.st_x
ohe = preprocessor.ohe

pd.set_option('display.max_columns',None)

#Function to display best parameters for RF by using GridSearchCV
def display(results):                                                           
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')
#classifier = RandomForestClassifier(n_estimators=50)
feature_list = list(preprocessor.x_one_train_df.columns)
#classifier_one = RandomForestClassifier(n_estimators=20, max_depth=8)
#cw = {1:}
classifier_one = RandomForestClassifier(n_estimators=50, max_depth=None, class_weight= "balanced")
'''
#Selecting best parameters

#rfc = RandomForestClassifier()
parameters = {
    "n_estimators":[5,10,50,100,250],
    "max_depth":[2,4,8,16,32,None]
    }
cv = GridSearchCV(classifier_one,parameters,cv=5)
cv.fit(x_one_train,y_train)
display(cv)'''
#classifier.fit(x_train,y_train)
classifier_one.fit(x_one_train,y_train)

#y_pred = classifier.predict(x_test)
y_one_pred = classifier_one.predict(x_one_test)

roc_auc = roc_auc_score(y_test, y_one_pred)
print("ROC_AUC Score for RF is " ,roc_auc)

'''# Get numerical feature importances
importances = list(classifier_one.feature_importances_)# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
# list of x locations for plotting
x_values = list(range(len(importances)))# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');'''
#one hot encoded
print("One Hot Encoded :- \n")
print(confusion_matrix(y_test,y_one_pred))
print(classification_report(y_test, y_one_pred))
print(accuracy_score(y_test, y_one_pred))
print("Mean Absolute Error = " , metrics.mean_absolute_error(y_test, y_one_pred))
print("Mean Squared Error = ", metrics.mean_squared_error(y_test, y_one_pred))
print("Root Mean Squared Error = ", np.sqrt(metrics.mean_squared_error(y_test, y_one_pred)))

#Saving model
filename_up = "rf.sav"
filename = "brain_stroke_predictor.sav"
#joblib.dump(classifier_one,filename)               #Uncomment to save new model
#joblib.dump(classifier_one,filename_up)             #upsampled classifier
#print decision tree
