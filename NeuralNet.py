# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:01:22 2023

@author: jerem
"""

'''
Implementing Neural Net
'''
from tensorflow.keras import models, layers, utils, backend as K
#import matplotlib.pyplot as plt
#import shap

n_features = 19 #onehotencoded

model = models.Sequential(name = "DeepNN", layers = [
    #hidden layer 1 (input)
    layers.Dense(
        name="h1",
        input_dim = 19,
        units = 1,
        activation = "relu"
        ),
    layers.Dropout(name="drop1", rate = 0.2),
    
    #hidden layer 2
    layers.Dense(
        name = "h2",
        units = int(round((n_features+1)/4)),
        activation = 'relu'
        ),
    layers.Dropout(name= "drop2", rate = 0.2),
    
    #output layer
    layers.Dense(name = "output", units = 1, activation = 'sigmoid')
    
    ])
model.summary()

model = models.model