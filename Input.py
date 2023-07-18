# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:56:23 2023

@author: jeremy
"""

import classify_new as cn
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import tkinter
from tkinter import *
from tkinter import filedialog


def openFile():
    filePath = filedialog.askopenfilename()
    data_set_new_rf = pd.read_csv(filePath)
    cn.ClassifyNew(data_set_new_rf)

window = Tk()
button = Button(text = "Open file", command = openFile)
button.pack()

window.mainloop()
