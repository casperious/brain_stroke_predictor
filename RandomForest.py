# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:00:42 2023

@author: jerem
"""

"""
Working on Random Forest Regression Model as per https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9962068/#B14-jcdd-10-00082
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
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

dummy = preprocessor.dummified

