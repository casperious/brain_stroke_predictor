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
'''app = customtkinter.CTk()
app.geometry("1200x800")
app.title("Brain Stroke Predictor")
'''
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
        cn.ClassifyNewRF(data_set_new_rf)
        cn.ClassifyNewSVM(data_set_new_svm)

#def close():
#    app.quit()

window = tk.Tk()
window.geometry("500x500")
window.title("Brain Stroke Predictor")

label = tk.Label(window,text = "Test", font = ('Arial',18))
label.pack(padx = 20, pady = 20)

button = tk.Button(text = "Open file", command = openFile)
button.pack()



'''
frame_1 = customtkinter.CTkFrame(master=app)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)

button = customtkinter.CTkButton(master=frame_1, text="Open File", command=openFile)
button.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)

quit_button = customtkinter.CTkButton(master=frame_1, text="Exit",command = close)
quit_button.place(relx = 1,rely = 0, anchor = customtkinter.NE)

textbox = customtkinter.CTkTextbox(frame_1)

app.mainloop()'''
window.mainloop()
