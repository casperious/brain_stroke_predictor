# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:56:23 2023

@author: jeremy
"""

import classify_new as cn
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import tkinter as tk
#from tkinter import *
from tkinter import filedialog
import customtkinter as customtkinter

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"
#customtkinter.set_default_color_theme("dark_blue")
app = customtkinter.CTk()
app.geometry("500x350")
app.title("Brain Stroke Predictor")

#CustomTKInter
frame = customtkinter.CTkFrame(master= app)
frame.pack(pady=20,padx = 60, fill = "both",expand = True)

label = customtkinter.CTkLabel(master= frame, text = "Brain Stroke Predictions", font = ("Arial",24))
label.pack(pady=12,padx=10)

rf_label = customtkinter.CTkLabel(master=frame, font=("Arial",16),text="Random Forest Prediction")
rf_label.pack(pady=12,padx=10)

svm_label= customtkinter.CTkLabel(master=frame, font=("Arial",16), text = "Support Vector Machine Prediction")
svm_label.pack(pady=12,padx=10)

'''

Input check. No empty string, has to be csv, formatted correctly

'''
def openFile():
    filePath = filedialog.askopenfilename()
    print(filePath)
    fileExtension = filePath[-4:]                   #grabs extension
    print(fileExtension)
    if(filePath!='' and fileExtension == ".csv"):   #only csv 
        data_set_new_rf = pd.read_csv(filePath)
        data_set_new_svm = pd.read_csv(filePath)
        prediction_rf = cn.ClassifyNewRF(data_set_new_rf)
        rf_label.configure(text= "Prediction by Random Forest is No stroke" if prediction_rf[0] == 0 else "Prediction by Random Forest is Stroke")
        prediction_svm = cn.ClassifyNewSVM(data_set_new_svm)
        svm_label.configure(text="Prediction by SVM is No stroke" if prediction_svm[0] == 0 else "Prediction by SVM is Stroke")

#def close():
#    app.quit()

button = customtkinter.CTkButton(master=frame,text="Open file", command = openFile)
button.pack(padx=10,pady=12)
#TKinter base
'''
window = tk.Tk()
window.geometry("500x500")
window.title("Brain Stroke Predictor")

label = tk.Label(window,text = "Test", font = ('Arial',18))
label.pack(padx = 20, pady = 20)


button = tk.Button(text = "Open file", command = openFile)
button.pack()

window.mainloop()
'''



app.mainloop()