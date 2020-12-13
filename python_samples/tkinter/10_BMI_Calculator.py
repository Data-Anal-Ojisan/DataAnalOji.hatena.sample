# -*- coding: utf-8 -*-
"""
@author: data-anal-ojisan
"""
import tkinter as tk


class App:

    def __init__(self):

        self.root = tk.Tk()                    # ウィンドウの呼び出し
        self.root.title('BMI Calculator')      # ウィンドウタイトル設定
        self.root.geometry('250x150')          # ウィンドウサイズ設定

        # 身長・体重・BMI指数に関連付けるtk.DoubleVar()をまとめたdictionary
        self.values = {'height': tk.DoubleVar(value=170.0),
                       'weight': tk.DoubleVar(value=60.0),
                       'bmi': tk.DoubleVar()}

    def launch(self):
        """
        アプリ起動メソッド
        """
        self.call_input_fields()            # 身長・体重の入力フィールドウィジェットを呼び出す
        self.call_calculate_button()        # BMI計算を実行するボタンwidgetを呼び出す
        self.call_result_field()            # 計算結果を表示するラベルウィジェットを呼び出す
        self.root.mainloop()                # アプリの起動状態を維持する

    def call_input_fields(self):
        """
        身長・体重の入力フィールドウィジェットを呼び出すメソッド
        """
        # ウィジェット配置のためのFrameを作成
        frame = tk.Frame(self.root, relief="ridge", bd=1)
        frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # 身長の入力フィールドを作成
        tk.Label(frame, text='身長 (cm)', width=10).grid(row=0, column=0, padx=5, pady=5)
        tk.Entry(frame, textvariable=self.values['height'], width=15).grid(row=0, column=1, padx=5, pady=5)

        # 体重の入力フィールドを作成
        tk.Label(frame, text='体重 (kg)', width=10).grid(row=1, column=0, padx=5, pady=5)
        tk.Entry(frame, textvariable=self.values['weight'], width=15).grid(row=1, column=1, padx=5, pady=5)

    def call_calculate_button(self):
        """
        BMI計算実行ボタンを呼び出すメソッド
        """
        # ボタンを作成
        tk.Button(self.root,
                  text='計算',
                  command=self.calculate).pack(fill=tk.X, padx=5, pady=5)

    def calculate(self):
        """
        BMI指数を計算するメソッド
        """
        # BMI指数を計算{体重(kg) / 身長(m)*身長(m)}
        bmi = self.values['weight'].get() / (self.values['height'].get() / 100)**2

        # 計算結果をtk.DoubleVar()に反映
        self.values['bmi'].set(bmi)

    def call_result_field(self):
        """
        BMI指数計算結果を表示するラベルを呼び出すメソッド
        """
        # ウィジェット配置のためのFrameを作成
        frame = tk.Frame(self.root, bg='white')
        frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # ラベルを作成
        tk.Label(frame, text='あなたのBMI指数', width=15, bg='white').pack(side=tk.LEFT)
        tk.Label(frame, textvariable=self.values['bmi'], bg='white').pack(side=tk.LEFT)


if __name__ == '__main__':

    app = App()
    app.launch()
