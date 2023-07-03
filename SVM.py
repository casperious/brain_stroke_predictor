#Data Pre-processing Stage
#importing libs

import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import category_encoders as ce
import sklearn as skl
from sklearn import preprocessing
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
import Preprocessor as preprocessor

#Variables for code from preprocessor
fit = preprocessor.fit
dummified = preprocessor.dummified_set
stroke_col = preprocessor.stroke_col
cols = preprocessor.cols
d= preprocessor.d
x_train = preprocessor.x_train
y_train = preprocessor.y_train
x_one_train = preprocessor.x_one_train
y_one_train = preprocessor.y_one_train
x_test = preprocessor.x_test
x_one_test = preprocessor.x_one_test
y_test = preprocessor.y_test
y_one_test = preprocessor.y_one_test
st_x = preprocessor.st_x

#SVM
from sklearn import svm #"Support Vector Classifier"
classifier = svm.SVC(kernel='rbf',gamma = 6,C = 2)
classifier_one = svm.SVC(kernel='rbf',gamma = 0.1,C=2)
classifier.fit(x_train,y_train)
classifier_one.fit(x_one_train,y_one_train)

#predicting test results
y_pred = classifier.predict(x_test)
y_one_pred = classifier_one.predict(x_one_test)

#printing report
print(classification_report(y_test,y_pred))
print(" -------------------------------------------------------------------------------------------------")
print(classification_report(y_one_test, y_one_pred))

#print("Where did the 100% come from")

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm_one = confusion_matrix(y_one_test, y_one_pred)

#plotting classificaton plain
#mtp.scatter()


#function to run classifier on new csv input
def ClassifyNew(data_set_new):
    # Using the dictionary to label future data
    #this should apply original encoding to new data
    for col in cols:
        print(data_set_new[col])
        data_set_new[col] = d[col].transform(data_set_new[col])
    
    print(data_set_new.to_string())
    x_new = data_set_new.iloc[:,data_set_new.columns!=stroke_col].values
    x_scaled = st_x.fit_transform(x_new)
    prediction = classifier.predict(x_scaled)
    print("Prediction by SVM is No stroke") if prediction[0] == 0 else print("Prediction is Stroke")
    