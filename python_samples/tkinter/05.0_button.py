# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:21:06 2020

@author: data-anal-ojisan
"""

import tkinter as ttk

root = ttk.Tk()

# window size
root.geometry('400x400')

# button
button = ttk.Button(text='push', width=20)
button.pack()

root.mainloop()