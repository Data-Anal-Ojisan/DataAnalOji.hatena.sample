# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:00:51 2020

@author: data-anal-ojisan
"""

import pickle
import tkinter as tk
from tkinter import ttk


class IrisClassifier:

    def __init__(self):

        self.root = tk.Tk()                                    # トップレベルウィンドウ
        self.classes = ['Setosa', 'Versicolour', 'Virginica']  # アヤメ種別のリスト

        # 事前学習済みモデルを読み込む
        with open('model/RandomForest_Iris.pickle', mode='rb') as fp:
            self.model = pickle.load(fp)

        # 分類結果を表示するためのtk.StringVar()
        self.predicted_class = tk.StringVar()

        # 入力フィールドの値をまとめたdictionary
        self.feature = {'sepal_length': tk.DoubleVar(value=3.0),
                        'sepal_width': tk.DoubleVar(value=3.0),
                        'petal_length': tk.DoubleVar(value=3.0),
                        'petal_width': tk.DoubleVar(value=3.0)}

    def launch(self):
        """
        アプリ起動用メソッド
        """

        self.call_window()                 # ウィンドウ設定を行う
        self.call_input_fields()           # 特徴量を指定する入力フィールドウィジェットを呼び出す
        self.call_classification_button()  # 分類処理を実行するボタンウィジェットを呼び出す
        self.call_result_label()           # 分類結果を表示するラベルウィジェットを呼び出す
        self.root.mainloop()               # アプリの起動状態を維持する

    def call_window(self):
        """
        ウィンドウの設定を行う
        """

        self.root.title('Iris Classification App')      # タイトルを変更する
        self.root.geometry('275x300')                   # ウィンドウサイズを指定する
        self.root.resizable(height=False, width=False)  # ウィンドウサイズ変更を不可にする

    def call_input_fields(self):
        """
        アヤメの特徴量を指定するための入力フィールドを呼び出す
        """

        # ウィジェット配置のためのLabelFrameを作成
        lf = ttk.LabelFrame(self.root, text='Features', padding=(10, 10))
        lf.pack(fill=tk.X, padx=5, pady=5)

        # 特徴量指定のための入力フィールドを作成
        for i, key in enumerate(self.feature.keys(), 0):
            tk.Label(lf, text=key, anchor='e', width=15).grid(row=i, column=0)
            tk.Label(lf, text=' : ').grid(row=i, column=1)
            tk.Entry(lf, textvariable=self.feature[key], justify='right', width=10).grid(row=i, column=2)

    def call_classification_button(self):
        """
        指定の特徴量をもとにアヤメ種別の分類処理を開始するボタンを呼び出す
        """

        tk.Button(self.root, text='Classify', command=self.classification).pack(fill=tk.X, padx=5, pady=5)

    def call_result_label(self):
        """
        分類結果を表示するラベルを呼び出す
        """

        # ウィジェット配置のためのFrameを作成
        f = tk.Frame(self.root, relief='solid', bd=1)
        f.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ラベルを作成
        tk.Label(f, text='Predicted class').pack(anchor='center')
        tk.Label(f,
                 textvariable=self.predicted_class,
                 bg='white',
                 font=('', 30),
                 foreground='#ff0000').pack(anchor='center', expand=True, fill=tk.BOTH)

    def classification(self):
        """
        アヤメの分類処理を実行し，分類結果を示すtk.StringVar()を更新するメソッド
        """

        # 入力データを作成する
        input_data = [[self.feature['sepal_length'].get(),
                       self.feature['sepal_width'].get(),
                       self.feature['petal_length'].get(),
                       self.feature['petal_width'].get()]]

        # 分類を行う
        predict = self.model.predict(input_data)

        # tk.StringVar()の更新
        self.predicted_class.set(self.classes[int(predict)])


if __name__ == '__main__':

    app = IrisClassifier()  # IrisClassifierのインスタンスを生成する
    app.launch()            # launch()メソッドでアプリを起動する
