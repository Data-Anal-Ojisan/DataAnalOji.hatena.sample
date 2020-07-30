# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:16:30 2020

@author: data-anal-ojisan
"""

import tkinter as tk

root = tk.Tk()

# window size
root.geometry('400x400')

# text box
textbox = tk.Entry(width=50)
textbox.pack()

# =============================================================================
# txtBox = tk.Entry()
# txtBox.configure(state='normal')   # writable
# txtBox.configure(state='readonly') # read only
# txtBox.pack()
# =============================================================================

root.mainloop()