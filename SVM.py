#Data Pre-processing Stage
#importing libs

import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import mlxtend as mlx
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
#importing data sets
data_set = pd.read_csv('brain_stroke.csv')

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#should switch to ordinal encoder, it will be able to handle unseen categorical data
#running into issue of new data being encoded with different values from training data
#label_encoder_x = LabelEncoder()
 #
#le = LabelEncoder()
#data_set[cols] = data_set[cols].apply(le.fit_transform)
#le_dict = dict(zip(le.classes_,le.transform(le.classes_)))      #trying to extract dictionary of values for encoded data
#encoding for dummy variables
#one_hot_encoder = OneHotEncoder()


cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
one_hot_encoded_data = pd.get_dummies(data_set, columns = cols,drop_first=True)

#extracting independent and dependent Variable
stroke_col = 'stroke'
x = one_hot_encoded_data.iloc[:,one_hot_encoded_data.columns!=stroke_col].values  #all columns excluding dependent variable
y = one_hot_encoded_data.iloc[:,5].values     #dependent variable

#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 4)

#feature scaling
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

#plot x_test scatter
mtp.scatter(x_test[:,0],x_test[:,1])

from sklearn import svm #"Support Vector Classifier"
classifier = svm.SVC(kernel='rbf',gamma = 0.1,C = 2)
classifier.fit(x_train,y_train)

#predicting test results
y_pred = classifier.predict(x_test)
y_train_pred = classifier.predict(x_train)

#printing report
from sklearn.metrics import classification_report
value = 0
width = 0.75
print(classification_report(y_test,y_pred))
print(classification_report(y_train, y_train_pred))
#plot_decision_regions(x_train,y_train,clf=classifier,
#                      filler_feature_values={2: value, 3: value, 4: value, 5: value,6:value,7:value,8:value,9:value,10:value,11:value,12:value,13:value,14:value,15:value,16:value,17:value,18:value},
#                      filler_feature_ranges={2: width, 3: width, 4: width, 5: width,6:width,7:width,8:width,9:width,10:width,11:width,12:width,13:width,14:width,15:width,16:width,17:width,18:width},
#                      legend=2)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#plotting classificaton plain
#mtp.scatter()

#function to run classifier on new csv input
def ClassifyNew(data_set_new):
    #data_set_new[cols] = data_set_new[cols].apply()#lambda x: le_dict.get(le.classes_))
    '''
    Running into issue where getting dummies for new data set, it creates dummy variables for only the data that shows up in the new data set
    not the dummies from the training data set which is what i want it mapped to
    '''
    new_hot_encoded_data = pd.get_dummies(data_set_new,columns = cols,drop_first=True)
    print(new_hot_encoded_data.to_string())
    x_new = new_hot_encoded_data.iloc[:,new_hot_encoded_data.columns!=stroke_col].values
    y_new = new_hot_encoded_data.iloc[:,5].values
    x_scaled = st_x.fit_transform(x_new)
    print(x_scaled)
    prediction = classifier.predict(x_scaled)
    print(prediction)