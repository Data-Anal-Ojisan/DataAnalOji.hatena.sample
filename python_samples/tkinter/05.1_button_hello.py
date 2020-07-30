# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:24:46 2020

@author: data-anal-ojisan
"""

import tkinter as ttk

root = ttk.Tk()

# window size
root.geometry('400x400')

def hello():
    print('Hello world!')

# button
button = ttk.Button(text='push', width=20, command=hello)
button.pack()

root.mainloop()