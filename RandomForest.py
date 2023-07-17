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
#import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.tree import export_graphviz
#from sklearn import tree
#Variables for code from preprocessor
data_set = preprocessor.data_set
#fit = preprocessor.fit
#dummified = preprocessor.dummified_set
stroke_col = preprocessor.stroke_col
cols = preprocessor.cols
#d= preprocessor.d
#d_o = preprocessor.d_o
x_train = preprocessor.x_train
y_train = preprocessor.y_train
x_one_train = preprocessor.x_one_train
#y_one_train = preprocessor.y_one_train
x_test = preprocessor.x_test
x_one_test = preprocessor.x_one_test
y_test = preprocessor.y_test
#y_one_test = preprocessor.y_one_test
st_x = preprocessor.st_x
#dummified_set = preprocessor.dummified_set
ohe = preprocessor.ohe

pd.set_option('display.max_columns',None)

#classifier = RandomForestClassifier(n_estimators=50)
classifier_one = RandomForestClassifier(n_estimators=20, random_state=1, max_depth=10)

#classifier.fit(x_train,y_train)
classifier_one.fit(x_one_train,y_train)

#y_pred = classifier.predict(x_test)
y_one_pred = classifier_one.predict(x_one_test)

#one hot encoded
print("One Hot Encoded :- \n")
print(confusion_matrix(y_test,y_one_pred))
print(classification_report(y_test, y_one_pred))
print(accuracy_score(y_test, y_one_pred))
print("Mean Absolute Error = " , metrics.mean_absolute_error(y_test, y_one_pred))
print("Mean Squared Error = ", metrics.mean_squared_error(y_test, y_one_pred))
print("Root Mean Squared Error = ", np.sqrt(metrics.mean_squared_error(y_test, y_one_pred)))

#print decision tree
'''fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
fn=data_set.columns
#cn=data_set.target_names
for index in range(0, 5):
    tree.plot_tree(classifier.estimators_[index],
                   feature_names = fn, 
                   #class_names=cn,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_5trees.png')'''
#function to run classifier on new csv input
def ClassifyNew(data_set_new):
    # The following code is for your newdf after training and testing on original df
    # Apply ohe on newdf
    cat_ohe_new = ohe.transform(data_set_new[preprocessor.categorical_cols])
    #Create a Pandas DataFrame of the hot encoded column
    ohe_df_new = pd.DataFrame(cat_ohe_new, columns = ohe.get_feature_names_out(input_features = preprocessor.categorical_cols))
    #concat with original data and drop original columns
    data_set_new.reset_index(drop=True,inplace=True)
    ohe_df_new.reset_index(drop=True,inplace=True)
    df_ohe_new = pd.concat([data_set_new, ohe_df_new], axis=1).drop(columns = preprocessor.categorical_cols, axis=1)
    #print(df_ohe_new.columns.to_list())
    print(df_ohe_new.head)
    
    df_ohe_new = st_x.transform(df_ohe_new)
    
    vals = classifier_one.predict_proba(df_ohe_new)
    #print(df_ohe_new)
    print("***********************")
    print(vals)
    print("*****************")
    # predict on df_ohe_new
    prediction = classifier_one.predict(df_ohe_new)
    
    '''print(data_set_new.columns.to_list())
    x_new = data_set_new.iloc[:,data_set_new.columns!=stroke_col].values
    print(x_new)
    x_scaled = st_x.fit_transform(x_new)
    prediction = classifier_one.predict(x_scaled)
    '''
    print("Prediction by Random Forest is No stroke") if prediction[0] == 0 else print("Prediction is Stroke")
