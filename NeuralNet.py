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

model = models.Sequential(name = "Perceptron", layers = [
    layers.Dense(
        name="dense",
        input_dim = 19,
        units = 1,
        activation = "linear"
        )
    ])
model.summary()