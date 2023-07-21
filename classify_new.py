# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:44:22 2023

@author: jerem
"""
import Preprocessor as preprocessor
import pandas as pd
import joblib
#function to run classifier on new csv input
def ClassifyNewRF(data_set_new):
    # The following code is for your newdf after training and testing on original df
    # Apply ohe on newdf
    cat_ohe_new = preprocessor.ohe.transform(data_set_new[preprocessor.categorical_cols])
    #Create a Pandas DataFrame of the hot encoded column
    ohe_df_new = pd.DataFrame(cat_ohe_new, columns = preprocessor.ohe.get_feature_names_out(input_features = preprocessor.categorical_cols))
    #concat with original data and drop original columns
    data_set_new.reset_index(drop=True,inplace=True)
    ohe_df_new.reset_index(drop=True,inplace=True)
    df_ohe_new = pd.concat([data_set_new, ohe_df_new], axis=1).drop(columns = preprocessor.categorical_cols, axis=1)
    #print(df_ohe_new.columns.to_list())
    #print(df_ohe_new.head)
    
    df_ohe_new = preprocessor.st_x.transform(df_ohe_new)
    
    #loading model
    #filename = "brain_stroke_predictor.sav"
    filename_up = "rf.sav"
    classifier_one = joblib.load(filename_up)
    
    vals = classifier_one.predict_proba(df_ohe_new)
    #print(df_ohe_new)
    print("***********************")
    print(vals)
    print("***********************")
    # predict on df_ohe_new
    prediction = classifier_one.predict(df_ohe_new)
    
    '''print(data_set_new.columns.to_list())
    x_new = data_set_new.iloc[:,data_set_new.columns!=stroke_col].values
    print(x_new)
    x_scaled = st_x.fit_transform(x_new)
    prediction = classifier_one.predict(x_scaled)
    '''
    print("Prediction by Random Forest is No stroke") if prediction[0] == 0 else print("Prediction by Random Forest is Stroke")

def ClassifyNewSVM(data_set_new):
    # The following code is for your newdf after training and testing on original df
    # Apply ohe on newdf
    cat_ohe_new = preprocessor.ohe.transform(data_set_new[preprocessor.categorical_cols])
    #Create a Pandas DataFrame of the hot encoded column
    ohe_df_new = pd.DataFrame(cat_ohe_new, columns = preprocessor.ohe.get_feature_names_out(input_features = preprocessor.categorical_cols))
    #concat with original data and drop original columns
    data_set_new.reset_index(drop=True,inplace=True)
    ohe_df_new.reset_index(drop=True,inplace=True)
    df_ohe_new = pd.concat([data_set_new, ohe_df_new], axis=1).drop(columns = preprocessor.categorical_cols, axis=1)
    #print(df_ohe_new.columns.to_list())
    #print(df_ohe_new.head)
    
    df_ohe_new = preprocessor.st_x.transform(df_ohe_new)
    
    #Loading Model
    filename = "svm_predictor.sav"
    filename_up = "svm_up.sav"
    classifier_one = joblib.load(filename_up)
    
    '''vals = classifier_one.predict_proba(df_ohe_new)
    #print(df_ohe_new)
    print("***********************")
    print(vals)
    print("***********************")'''
    # predict on df_ohe_new
    prediction = classifier_one.predict(df_ohe_new)
    print("Prediction by SVM is No stroke") if prediction[0] == 0 else print("Prediction by SVM is Stroke")