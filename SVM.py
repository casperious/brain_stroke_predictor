import numpy as np
#import matplotlib.pyplot as mtp
import pandas as pd
#import category_encoders as ce
import sklearn as skl
from sklearn import preprocessing
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.compose import ColumnTransformer
import Preprocessor as preprocessor
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
#Variables for code from preprocessor
#fit = preprocessor.fit
#dummified = preprocessor.dummified_set
stroke_col = preprocessor.stroke_col
cols = preprocessor.cols
#d= preprocessor.d
x_train = preprocessor.x_train
y_train = preprocessor.y_train
x_one_train = preprocessor.x_one_train
#y_one_train = preprocessor.y_one_train
x_test = preprocessor.x_test
x_one_test = preprocessor.x_one_test
y_test = preprocessor.y_test
#y_one_test = preprocessor.y_one_test
st_x = preprocessor.st_x
ohe = preprocessor.ohe
#SVM
from sklearn import svm #"Support Vector Classifier"
#classifier = svm.SVC(kernel='rbf',gamma = 6,C = 2)
classifier_one = svm.SVC(gamma = 1,C=5,probability=True) #kernel='rbf',
#classifier.fit(x_train,y_train)
classifier_one.fit(x_one_train,y_train)

'''
#checking optimal hyperparams
param_grid = {'C': [0.1,1,5], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
grid.fit(x_one_train,y_train)
print(grid.best_estimator_)
'''
#predicting test results
#y_pred = classifier.predict(x_test)
y_one_pred = classifier_one.predict(x_one_test)

roc_auc = roc_auc_score(y_test, y_one_pred)
print("ROC_AUC Score for SVM is " ,roc_auc, "\n")

report = classification_report(y_test, y_one_pred)
accuracy = accuracy_score(y_test,y_one_pred)
saved_accuracy_score = 101                                  #have to figure out a way to save highest accuracy with file

print(confusion_matrix(y_test,y_one_pred))
print(report)
print(accuracy_score(y_test, y_one_pred))
print("Mean Absolute Error = " , metrics.mean_absolute_error(y_test, y_one_pred))
print("Mean Squared Error = ", metrics.mean_squared_error(y_test, y_one_pred))
print("Root Mean Squared Error = ", np.sqrt(metrics.mean_squared_error(y_test, y_one_pred)))

#Saving model
filename = "svm_predictor.sav"
filename_up = "svm_up.sav"
if(saved_accuracy_score<accuracy):    
    joblib.dump(classifier_one,filename_up)

#plotting classificaton plain
#mtp.scatter()
