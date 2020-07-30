# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:35:33 2020

@author: data-anal-ojisan
"""

import tkinter as ttk

root = ttk.Tk()

# window size
root.geometry('400x400')

# frame
frame = ttk.Frame(
    root, width=200, height=100,
    borderwidth=10, relief='ridge',
    bg='blue', bd=10)
frame.pack()

root.mainloop()