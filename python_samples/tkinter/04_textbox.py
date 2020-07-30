# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:16:30 2020

@author: data-anal-ojisan
"""

import tkinter as ttk

root = ttk.Tk()

# window size
root.geometry('400x400')

# text box
textbox = ttk.Entry(width=50)
textbox.pack()

# =============================================================================
# txtBox = ttk.Entry()
# txtBox.configure(state='normal')   # writable
# txtBox.configure(state='readonly') # read only
# txtBox.pack()
# =============================================================================

root.mainloop()