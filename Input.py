# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:56:23 2023

@author: jeremy
"""

import SVM as svm
import RandomForest as rf
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import tkinter
from tkinter import *
from tkinter import filedialog

#data_set_new = pd.read_csv('new_patient_data.csv')

def openFile():
    filePath = filedialog.askopenfilename()
    #print(filePath)
    data_set_new_svm = pd.read_csv(filePath)
    data_set_new_rf = pd.read_csv(filePath)
    #print(data_set_new_rf.type)
    svm.ClassifyNew(data_set_new_svm)
    rf.ClassifyNew(data_set_new_rf)
    

#accept new csv file to read and predict
window = Tk()
button = Button(text = "Open file", command = openFile)
button.pack()

window.mainloop()
