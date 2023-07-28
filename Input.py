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
app.title("Brain Stroke Predictor")

#background_image = ImageTk.PhotoImage(Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/background.jpeg").resize((2500, 2500)))
background_image = customtkinter.CTkImage(light_image=Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/bg3.jpg"),
                                          dark_image = Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/bg3.jpg")
                                          , size=(2500,2500))

#CustomTKInter



frame = customtkinter.CTkFrame(master= app,fg_color="transparent")
frame.pack(pady=20,padx = 60, fill = "both",expand = True)
#frame.configure('-alpha', 0.25)
bg_label = customtkinter.CTkLabel(master=frame, image=background_image,text="")
bg_label.place(relx=0.5,rely=0.5,anchor='center')

top_frame = customtkinter.CTkFrame(master=frame)
top_frame.pack(padx=10,pady=12,side=customtkinter.TOP)

data_frame = customtkinter.CTkFrame(master=frame)
data_frame.pack(padx=0,pady=10,side=customtkinter.LEFT)

#data_bg_image = customtkinter.CTkLabel(master=data_frame)#, image=bg_image_transparent)
#data_bg_image.place(relx=0.5,rely=0.5,anchor='center')

image_frame = customtkinter.CTkFrame(master=frame)
image_frame.pack(padx=10,pady=10,expand=True,side=customtkinter.RIGHT)

label = customtkinter.CTkLabel(master= top_frame, text = "Brain Stroke Predictions", font = ("Arial",24))
label.pack(pady=12,padx=10)

rf_label = customtkinter.CTkLabel(master=data_frame, font=("Arial",16),text="Random Forest Prediction",wraplength=250)
rf_label.pack(pady=12,padx=10)

svm_label= customtkinter.CTkLabel(master=data_frame, font=("Arial",16), text = "Support Vector Machine Prediction",wraplength=250)
svm_label.pack(pady=12,padx=10)

current_new_data = customtkinter.CTkLabel(master=data_frame,text="Current Patient Data is", font = ("Arial",16))
current_new_data.pack(pady=12,padx=10)

current_data_frame = customtkinter.CTkFrame(master=data_frame)
current_data_frame.pack(padx=10,pady=12)


columns = list(data.columns)
for column in columns:
    new_column = customtkinter.CTkTextbox(master = current_data_frame,width=76,height=50,bg_color='transparent')
    new_column.insert("0.0", column)
    new_column.configure(state='disabled')
    new_column.pack(side=customtkinter.LEFT,padx=5,pady=12)

#either write new input to og csv, or create a new data frame and save new input into it
def writeData(entry):
    print("writing ",entry)
    


current_data = customtkinter.CTkFrame(master=data_frame)
current_data.pack(padx=10,pady=10)
new_data = data.iloc[0]
for entry in new_data:
    new_column = customtkinter.CTkTextbox(master = current_data,width=76,height=50)
    new_column.insert("0.0",entry)
    #new_column.bind("<Enter>",lambda x: writeData(entry))                                   #Have to figure out how to save changes
    new_column.pack(side=customtkinter.LEFT,padx=5,pady=12)
    
    
'''video
video_player = TkinterVideo(master = image_frame,scaled=True)
video_player.load("C:/Users/jerem/Software/Projects/Support Vector Machine Training/stroke.mp4")
video_player.pack(expand=True)
video_player.play()
'''

my_image = customtkinter.CTkImage(light_image=Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/TIA.png"),
                                  dark_image=Image.open("C:/Users/jerem/Software/Projects/Support Vector Machine Training/TIA.png"),
                                  size=(700, 500))

image_label = customtkinter.CTkLabel(master=image_frame, image=my_image, text="")  # display image with a CTkLabel
image_label.pack(padx=10,pady=12)

def callback(url):
   webbrowser.open_new_tab(url)

link_frame = customtkinter.CTkFrame(master=image_frame)
link_frame.pack(padx=10,pady=12,expand=True, fill="both")

link_button = customtkinter.CTkButton(master=link_frame, text = "Research Paper on Strokes", cursor="hand2", command =lambda: callback("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7589849/"))
link_button.pack(padx=10,pady=12)
    
link_button_2 = customtkinter.CTkButton(master=link_frame, text = "Brain Stroke Data Repository", cursor="hand2", command =lambda: callback("https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset"))
link_button_2.pack(padx=10,pady=12)

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


button = customtkinter.CTkButton(master=data_frame,text="Open file", command = openFile)
button.pack(padx=10,pady=12)

#app.attributes('-fullscreen',True)
app.mainloop()