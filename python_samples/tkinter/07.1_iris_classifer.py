# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:00:51 2020

@author: data-anal-ojisan
"""

import os
import pickle
import numpy as np
import tkinter as ttk

root = ttk.Tk()

# title
root.title('iris classifer')

# window size
root.geometry('500x400')


Static1 = ttk.Label(text=u'sepal_length') # ラベル
Static1.pack()
EditBox1 = ttk.Entry() # 1行入力ボックス
EditBox1.pack()
 
Static2 = ttk.Label(text=u'sepal_width')
Static2.pack()
EditBox2 = ttk.Entry()
EditBox2.pack()
 
Static3 = ttk.Label(text=u'petal_length')
Static3.pack()
EditBox3 = ttk.Entry()
EditBox3.pack()
 
Static4 = ttk.Label(text=u'petal_width')
Static4.pack()
EditBox4 = ttk.Entry()
EditBox4.pack()

def dlModel(event):
    global predict
    ret = True
    
    if ret:
        val1 = float(EditBox1.get())
        val2 = float(EditBox2.get())
        val3 = float(EditBox3.get())
        val4 = float(EditBox4.get())
        
        test_data = [val1,val2,val3,val4]
        test_data = np.array(test_data).reshape(1,-1)
 
        with open('RandomForest_Iris.pickle', mode='rb') as fp:
                model = pickle.load(fp)
        predict = model.predict(test_data)
        
        # # 最も値が大きいものを出力
        if predict == 0:
            label["text"] = "Setosa"
        elif predict == 1:
            label["text"] = "Versicolour"
        else:
            label["text"] = "Virginica"
            
# Predict
Button = ttk.Button(text=u'Predict', font=8)
Button.bind("<Button-1>", dlModel) # <Button-1>は左クリック、クリックするとdlModelが呼び出される
Button.pack(pady=5)
 
# 結果
label = ttk.Label(text=u' ', font=8, foreground='#ff0000')
label.pack(pady=5)
 


root.mainloop()