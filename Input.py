# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:56:23 2023

@author: jeremy
"""

'''
UI Improvements, different blocks for data entry where users can change input variable values. Heart Diagram with links, scroll page
for 

Think about live data feed

'''


import classify_new as cn
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import tkinter as tk
#from tkinter import *
from tkinter import filedialog
import customtkinter as customtkinter
from PIL import Image, ImageTk
import webbrowser



data = pd.read_csv('new_patient_data.csv')
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"
#customtkinter.set_default_color_theme("dark_blue")
app = customtkinter.CTk()
#app.geometry("500x350")
app.geometry("{0}x{1}+0+0".format(app.winfo_screenwidth(), app.winfo_screenheight()))
#app.geometry("1920x1280")
app.title("Quick Thinking")
app.columnconfigure(0,weight = 0)
app.columnconfigure(1,weight = 1)
app.grid_rowconfigure(0, weight = 0)
app.grid_rowconfigure(1, weight = 1)

#background_image = ImageTk.PhotoImage(Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/background.jpeg").resize((2500, 2500)))
background_image = customtkinter.CTkImage(light_image=Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/bg3.jpg"),
                                          dark_image = Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/bg3.jpg")
                                          , size=(4000,4000))

#CustomTKInter
frame = customtkinter.CTkFrame(master= app,fg_color="transparent")
frame.pack(pady=20,padx = 60, fill = "both",expand = True)
#frame.configure('-alpha', 0.25)
bg_label = customtkinter.CTkLabel(master=frame, image=background_image,text="")
bg_label.place(relx=0.5,rely=0.5,anchor='center',relwidth=1,relheight=1)


top_frame = customtkinter.CTkFrame(master=frame)
top_frame.grid(padx=10,pady=12,column=1,row=0,sticky="nsew")

data_frame = customtkinter.CTkFrame(master=frame)
data_frame.grid(padx=0,pady=5,column=0,row=1,sticky="n")

image_frame = customtkinter.CTkFrame(master=frame)
image_frame.grid(padx=10,pady=10,column=2,row=1,sticky="n")

label = customtkinter.CTkLabel(master= top_frame, text = "Quick Thinking", font = ("Arial",24))
label.grid(pady=12,padx=10,sticky="n")

rf_label = customtkinter.CTkLabel(master=data_frame, font=("Arial",16),text="Random Forest Prediction",wraplength=250)
rf_label.pack(pady=5,padx=5)

svm_label= customtkinter.CTkLabel(master=data_frame, font=("Arial",16), text = "Support Vector Machine Prediction",wraplength=250)
svm_label.pack(pady=5,padx=5)

current_new_data = customtkinter.CTkLabel(master=data_frame,text="Current Patient Data is", font = ("Arial",16))
current_new_data.pack(pady=5,padx=10)

current_data_frame = customtkinter.CTkFrame(master=data_frame)
current_data_frame.pack(padx=10,pady=5)


columns = list(data.columns)
for column in columns:
    new_column = customtkinter.CTkTextbox(master = current_data_frame,width=75,height=50,bg_color='transparent',font=("Arial",12))
    new_column.insert("0.0", column)
    new_column.configure(state='disabled')
    new_column.pack(side=customtkinter.LEFT,padx=5,pady=5)

#either write new input to og csv, or create a new data frame and save new input into it
def writeData(entry):
    print("writing ",entry)
    


current_data = customtkinter.CTkFrame(master=data_frame)
current_data.pack(padx=10,pady=5)

new_data = data.iloc[0]
for entry in new_data:
    new_column = customtkinter.CTkTextbox(master = current_data,width=75,height=50,font=("Arial",12))
    new_column.insert("0.0",entry)
    #new_column.bind("<Enter>",lambda x: writeData(entry))                                   #Have to figure out how to save changes
    new_column.pack(side=customtkinter.LEFT,padx=5,pady=5)
    

my_image = customtkinter.CTkImage(light_image=Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/TIA.png"),
                                  dark_image=Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/TIA.png"),
                                  size=(500, 300))

image_label = customtkinter.CTkLabel(master=image_frame, image=my_image, text="")  # display image with a CTkLabel
image_label.pack(padx=10,pady=5)

image_text = customtkinter.CTkLabel(master=image_frame, text="Brain Stroke Causes",font = ("Arial",12))
image_text.pack(padx=10,pady=5)


mri_frame = customtkinter.CTkFrame(master=frame)
mri_frame.grid(padx=0,pady=5,column=0,row=2,sticky='n')

mri_image = customtkinter.CTkImage(light_image=Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/mri.jpg"),
                                   dark_image=Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/mri.jpg"),
                                   size=(500,500))
mri_label = customtkinter.CTkLabel(master=mri_frame,image=mri_image,text="")
mri_label.pack(padx=10,pady=5)

mri_text = customtkinter.CTkLabel(master=mri_frame, text="MRI Scans", font=("Arial",12))
mri_text.pack(padx=10,pady=5)

def callback(url):
   webbrowser.open_new_tab(url)

link_frame = customtkinter.CTkFrame(master=frame)
link_frame.grid(padx=10,pady=12,column=2,row=2,sticky='n')

link_button = customtkinter.CTkButton(master=link_frame, text = "Research Paper on Strokes", cursor="hand2", command =lambda: callback("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7589849/"))
link_button.pack(padx=10,pady=12)
    
link_button_2 = customtkinter.CTkButton(master=link_frame, text = "Brain Stroke Data Repository", cursor="hand2", command =lambda: callback("https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset"))
link_button_2.pack(padx=10,pady=12)

notes_text = customtkinter.CTkTextbox(master=link_frame,width=500,height=450)
lines = open("Notes.txt",'r')
notes_text.insert("0.0",lines.read())
lines.close()
notes_text.pack(padx=5,pady=5)

'''

Input check. No empty string, has to be csv, formatted correctly

'''
def openFile():
    filePath = filedialog.askopenfilename()
    #print(filePath)
    fileExtension = filePath[-4:]                   #grabs extension
    #print(fileExtension)
    if(filePath!='' and fileExtension == ".csv"):   #only csv 
        data_set_new_rf = pd.read_csv(filePath)
        data_set_new_svm = pd.read_csv(filePath)
        prediction_rf,vals_rf = cn.ClassifyNewRF(data_set_new_rf)
        #print(vals_rf)
        if(prediction_rf[0]==0):
            rf_label.configure(text="Prediction with Random Forest is No Stroke with "+str(round(vals_rf[0][0]*100,2))+"% probability")
        else:
            rf_label.configure(text= "Prediction by Random Forest is Stroke with "+str(round(vals_rf[0][1]*100,2))+"% probability")
        
        prediction_svm,vals_svm = cn.ClassifyNewSVM(data_set_new_svm)
        if(prediction_svm[0]==0):
            svm_label.configure(text="Prediction with SVM is No Stroke with "+str(round(vals_svm[0][0]*100,2))+"% probability")      
        else:
            svm_label.configure(text="Prediction by SVM is Stroke with "+str(round(vals_svm[0][1]*100,2))+"% probability")


button = customtkinter.CTkButton(master=data_frame,text="Predict New Case", command = openFile)
button.pack(padx=10,pady=12)

#app.attributes('-fullscreen',True)
app.mainloop()