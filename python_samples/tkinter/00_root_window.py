# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:30:51 2020

@author: data-anal-ojisan
"""

import tkinter as tk

# 基本形
root = tk.Tk()   # ルートウィンドウ（Top-level widget) を作成する
root.mainloop()  # アプリケーション起動を維持

# タイトル変更
root = tk.Tk()
root.title('SampleApp')  # アプリケーションタイトルを変更する
root.mainloop()

# ウィンドウサイズ変更
root = tk.Tk()
root.geometry('400x400')  # ウィンドウサイズを指定する
root.mainloop()

# ウィンドウサイズ固定
root = tk.Tk()
root.resizable(height=False, width=False)  # ウィンドウサイズを固定する
root.mainloop()

# 最小ウィンドウサイズ指定
root = tk.Tk()
root.minsize(height=200, width=200)  # 最小ウィンドウサイズを指定する
root.mainloop()

# 最大ウィンドウサイズ指定
root = tk.Tk()
root.maxsize(height=200, width=200)  # 最大ウィンドウサイズを指定する
root.mainloop()

# 背景色変更
root = tk.Tk()
root.configure(background='#FFF000000')  # 背景色を指定する
root.mainloop()

# ウィンドウ枠追加
root = tk.Tk()
root.configure(relief='sunken', bd=10)  # ウィンドウ枠を追加する
root.mainloop()

# アイコン変更
root = tk.Tk()
icon_path = 'common/python.ico'     # *.ico ファイルのパス
root.iconbitmap(default=icon_path)  # アイコンを変更する
root.mainloop()
