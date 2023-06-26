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

#importing data sets
data_set = pd.read_csv('brain_stroke.csv')

stroke_col = 'stroke'
cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
col_indices = [0,4,5,6,9]

#encoding categorical data
d = defaultdict(LabelEncoder)
# Encoding the variable
fit = data_set.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))

#extracting independent and dependent Variable
x = fit.iloc[:,fit.columns!=stroke_col].values  #all columns excluding dependent variable
y = fit.iloc[:,10].values     #dependent variable

#splitting dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 4)

#feature scaling
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

from sklearn import svm #"Support Vector Classifier"
classifier = svm.SVC(kernel='rbf',gamma = 6,C = 2)
classifier.fit(x_train,y_train)

#predicting test results
y_pred = classifier.predict(x_test)

#printing report
print(classification_report(y_test,y_pred))

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#plotting classificaton plain
#mtp.scatter()

#function to run classifier on new csv input
def ClassifyNew(data_set_new):
    # Using the dictionary to label future data
    #this should apply original encoding to new data
    for col in cols:
        data_set_new[col] = d[col].transform(data_set_new[col])
    
    print(data_set_new.to_string())
    x_new = data_set_new.iloc[:,data_set_new.columns!=stroke_col].values
    x_scaled = st_x.fit_transform(x_new)
    prediction = classifier.predict(x_scaled)
    print("Prediction is No stroke") if prediction[0] == 0 else print("Prediction is Stroke")
    