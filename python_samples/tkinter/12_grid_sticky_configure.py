import tkinter as tk



def fill_one_column(self):
    root = tk.Tk()
    root.geometry('500x300')
    root.title('Fill one column')
    tk.Label(root, text='fill=tk.X', bg='green').pack(fill=tk.X)
    root.mainloop()

def sticky_one_column(self):
    root = tk.Tk()
    root.geometry('500x300')
    root.title('Sticky one column')
    tk.Label(root, text='tk.E+tk.W', bg='green').grid(row=0, column=0, sticky=tk.E+tk.W)
    root.mainloop()

def sticky_two_columns(self):
    root = tk.Tk()
    root.geometry('500x300')
    root.title('Sticky two columns')
    tk.Label(root, text='Label1, sticky=tk.EW', bg='green').grid(row=0, column=0, sticky=tk.EW)
    tk.Label(root, text='Label2, sticky=tk.EW', bg='red').grid(row=0, column=1, sticky=tk.EW)
    root.mainloop()

def sticky_one_column_with_configure(self):
    root = tk.Tk()
    root.geometry('500x300')
    root.grid_columnconfigure(0, weight=1)
    root.title('Sticky one column with configure')
    tk.Label(root, text='sticky=tk.EW with configure', bg='green').grid(row=0, column=0, sticky=tk.EW)
    root.mainloop()

def sticky_two_columns_with_configure(self):
    root = tk.Tk()
    root.geometry('500x300')
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.title('Sticky two columns with configure')
    tk.Label(root, text='Label1, sticky=tk.EW with configure', bg='green').grid(row=0, column=0, sticky=tk.EW)
    tk.Label(root, text='Label2, sticky=tk.EW with configure', bg='red').grid(row=0, column=1, sticky=tk.EW)
    root.mainloop()

def sticky_two_rows_with_configure(self):
    root = tk.Tk()
    root.geometry('500x300')
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.title('Sticky two rows with configure')
    tk.Label(root, text='Label1, sticky=tk.EW with configure', bg='green').grid(row=0, column=0, sticky=tk.NS)
    tk.Label(root, text='Label2, sticky=tk.EW with configure', bg='red').grid(row=1, column=0, sticky=tk.NS)
    root.mainloop()


if __name__ == '__main__':

    root = tk.Tk()
    root.geometry('300x400')
    root.columnconfigure(0, weight=2)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=3)
    root.rowconfigure(1, weight=1)
    tk.Label(root, text='row=0, column=0', bg='green', fg='white').grid(row=0, column=0, sticky=tk.NSEW)
    tk.Label(root, text='row=0, column=1', bg='red', fg='white').grid(row=0, column=1, sticky=tk.NSEW)
    tk.Label(root, text='row=1, column=0', bg='blue', fg='white').grid(row=1, column=0, sticky=tk.NSEW)
    tk.Label(root, text='row=1, column=1', bg='yellow', fg='black').grid(row=1, column=1, sticky=tk.NSEW)
    root.mainloop()