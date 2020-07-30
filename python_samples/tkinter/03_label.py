# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:15:15 2020

@author: rhira
"""

import tkinter as tk

root = tk.Tk()

# window size
root.geometry('400x400')

# label
label = tk.Label(root, text='sample')
label.pack()

root.mainloop()