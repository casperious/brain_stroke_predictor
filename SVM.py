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
#importing data sets
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
#fit_one = data_set.apply(lambda x: d[x.name].fit_transform(x))
#fit_one = pd.DataFrame(ohe.fit_transform(data_set))
#fit_one = pd.get_dummies(data_set, drop_first = True)
# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))
#fit_one.apply(lambda x: d[x.name].inverse_transform(x))


'''
One Hot Encoding with dummy vars

'''

def dummify(OHE, x, columns):
    transformed_array = OHE.transform(x)
    initial_colnames_keep = list(set(x.columns.tolist()) - set(columns))
    new_colnames = nm.concatenate(model_OHE.named_transformers_['OHE'].categories_).tolist()
    all_colnames = new_colnames + initial_colnames_keep 
    df = pd.DataFrame(transformed_array, index = x.index, columns = all_colnames)
    return df

model_OHE = ColumnTransformer(
    [('OHE', OneHotEncoder(),['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])],
    remainder = 'passthrough'
    )
dummified = model_OHE.fit(data_set)
dummified = dummify(model_OHE, data_set,cols)

'''
Below is independent and dependent split of label encoded data_set
'''
#extracting independent and dependent Variable
x = fit.iloc[:,fit.columns!=stroke_col].values  #all columns excluding dependent variable
y = fit.iloc[:,10].values     #dependent variable

'''
Below is independent and dependent split of one hot encoded dummy var data set
'''
x_one = dummified.iloc[:,dummified.columns!=stroke_col].values
y_one = dummified.iloc[:,19].values


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

print("Where did the 100% come from")

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
        data_set_new[col] = d[col].transform(data_set_new[col])
    
    print(data_set_new.to_string())
    x_new = data_set_new.iloc[:,data_set_new.columns!=stroke_col].values
    x_scaled = st_x.fit_transform(x_new)
    prediction = classifier.predict(x_scaled)
    print("Prediction is No stroke") if prediction[0] == 0 else print("Prediction is Stroke")
    