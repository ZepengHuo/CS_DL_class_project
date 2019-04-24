#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tkinter

from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tkinter import ttk
import numpy


top = Tk()
top.title = 'cifar10'
top.geometry('1000x650')
canvas = Canvas(top, width=560,height=560, bd=0,bg='white')
canvas.grid(sticky="W", row=3)

def showImg():
    '''
	File = askopenfilename(title='Open Image')

    e.set(File)

    load = Image.open(e.get())
    w, h = load.size
    load = load.resize((1*w, 1*h))
    imgfile = ImageTk.PhotoImage(load )

    canvas.image = imgfile  # <--- keep reference of your image
    canvas.create_image(20,20,anchor='nw',image=imgfile)
	'''
    frame.grid(row=1, column=0)
	
	
e = StringVar()

submit_button = Button(top, text ="Generate test patient's data", command = showImg)
submit_button.grid(row=0, column=0)

label_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def Predict():

    pred = e.get().split('/')[-1][:-4]




    #textvar = "The prediction is : '%s' image" %(pred)
    textvar = "The prediction is : \n P1: readmission in 72 hours \n P2 readmission in 24 hours \n P3: non-readmission \n P4: readmission in 7 days \n P5: readmission in 30 days" 
	
	
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar+'\n')
    t1.update()
	
    textvar_2 = "The ground truth is :  \n \u2717 P1: readmission in 48 hours \n \u2713 P2 readmission in 24 hours \n \u2713 P3: non-readmission \n \u2717 P4: readmission in 24 \n \u2713 P5: readmission in 30 days" 
	
	
    t2.delete(0.0, tkinter.END)
    t2.insert('insert', textvar_2+'\n')
    t2.update()
	
def show_AUC():
	# show AUC
    load = Image.open('AUROC.PNG')
    w, h = load.size
    load = load.resize((1*w, 1*h))
    imgfile = ImageTk.PhotoImage(load )

    canvas.image = imgfile  # <--- keep reference of your image
    canvas.create_image(20,20,anchor='nw',image=imgfile)
	
	
submit_button = Button(top, text ='Predict', command = Predict)
submit_button.grid(row=0, column=1)

result_button = Button(top, text ='Show result', command = show_AUC)
result_button.grid(row=2, column=1)

l1=Label(top,text='Plot of AUROC results')
l1.grid(row=2, column=0)

#l2=Label(top,text='Showing AUROC evaluation')
#l2.grid(row=0, column=3)

t1=Text(top,bd=0, width=27,height=10,font='Fixdsys -14')
t1.grid(row=1, column=1)

t2=Text(top,bd=0, width=27,height=10,font='Fixdsys -14')
t2.grid(row=1, column=2)

#########################################################################################################
data = [ 
         ["1", "87", "120/77"],
         ["1", "66", "113/88"],
         ["3", "110", "130/98"],
         ["4", "72", "141/99"],
         ["5", "99", "135/89"] ]

frame = Frame(top)


tree = ttk.Treeview(frame, columns = (1,2,3), height = 5, show = "headings")
tree.pack(side = 'left')

tree.heading(1, text="Patient ID")
tree.heading(2, text="Heart Rate")
tree.heading(3, text="Blodd Pressure")

tree.column(1, width = 100)
tree.column(2, width = 100)
tree.column(3, width = 100)

scroll_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
scroll_y.pack(side = 'right', fill = 'y')


scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
scroll_x.pack(side = BOTTOM, fill = 'x')

tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

for val in data:
    tree.insert('', 'end', values = (val[0], val[1], val[2]) )


	
###############################################################################################################
top.mainloop()
