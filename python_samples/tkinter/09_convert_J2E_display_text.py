# -*- coding: utf-8 -*-
"""
@author: data-anal-ojisan
"""

import tkinter as tk
from language import english, japanese


class App:

    def __init__(self):

        self.root = tk.Tk()                    # ウィンドウの呼び出し
        self.root.title('Sample Application')  # ウィンドウタイトル設定
        self.root.geometry('250x180')          # ウィンドウサイズ設定

        # GUI表示に関連付けるtk.StringVar()をまとめたdictionary
        self.display_text = {'height': tk.StringVar(value='Height (cm)'),
                             'weight': tk.StringVar(value='Weight (kg)'),
                             'calculate': tk.StringVar(value='Calculate'),
                             'result': tk.StringVar(value='Your BMI index is')}

        # 身長・体重・BMI指数に関連付けるtk.DoubleVar()をまとめたdictionary
        self.values = {'height': tk.DoubleVar(value=170.0),
                       'weight': tk.DoubleVar(value=60.0),
                       'bmi': tk.DoubleVar()}

    def launch(self):
        """
        アプリ起動メソッド
        """
        self.call_switch_language_button()  # 表示言語切り替えボタンウィジェットを呼び出す
        self.call_input_fields()            # 身長・体重の入力フィールドウィジェットを呼び出す
        self.call_calculate_button()        # BMI計算を実行するボタンwidgetを呼び出す
        self.call_result_field()            # 計算結果を表示するラベルウィジェットを呼び出す
        self.root.mainloop()                # アプリの起動状態を維持する

    def call_switch_language_button(self):
        """
        表示テキストの日⇔英切り替えボタンウィジェットを呼び出すメソッド
        """
        # ウィジェット配置のためのFrameを作成
        frame = tk.Frame(self.root, relief="ridge", bd=1)
        frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # 日本語表示に切り替えるボタンウィジェットの呼び出し
        tk.Button(master=frame,
                  text='日本語',
                  command=lambda: self.switch_language(japanese.text)).pack(fill=tk.X, expand=True, side=tk.LEFT)

        # 英語表示に切り替えるボタンウィジェットの呼び出し
        tk.Button(master=frame,
                  text='English',
                  command=lambda: self.switch_language(english.text)).pack(fill=tk.X, expand=True, side=tk.LEFT)

    def switch_language(self, text):
        """
        表示言語切り替えを実行するメソッド
        """
        # forループ内でtk.StringVar()の値を変更
        for key in self.display_text.keys():
            self.display_text[key].set(text[key])

    def call_input_fields(self):
        """
        身長・体重の入力フィールドウィジェットを呼び出すメソッド
        """
        # ウィジェット配置のためのFrameを作成
        frame = tk.Frame(self.root, relief="ridge", bd=1)
        frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # 身長の入力フィールドを作成
        tk.Label(frame, textvariable=self.display_text['height'], width=10).grid(row=0, column=0, padx=5, pady=5)
        tk.Entry(frame, textvariable=self.values['height'], width=15).grid(row=0, column=1, padx=5, pady=5)

        # 体重の入力フィールドを作成
        tk.Label(frame, textvariable=self.display_text['weight'], width=10).grid(row=1, column=0, padx=5, pady=5)
        tk.Entry(frame, textvariable=self.values['weight'], width=15).grid(row=1, column=1, padx=5, pady=5)

    def call_calculate_button(self):
        """
        BMI計算実行ボタンを呼び出すメソッド
        """
        # ボタンを作成
        tk.Button(self.root,
                  textvariable=self.display_text['calculate'],
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
        tk.Label(frame, textvariable=self.display_text['result'], width=15, bg='white').pack(side=tk.LEFT)
        tk.Label(frame, textvariable=self.values['bmi'], bg='white').pack(side=tk.LEFT)


if __name__ == '__main__':

    app = App()
    app.launch()
