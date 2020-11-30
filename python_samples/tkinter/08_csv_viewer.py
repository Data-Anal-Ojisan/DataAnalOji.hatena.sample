# -*- coding: utf-8 -*-
"""
@author: data-anal-ojisan
"""

import os
import csv
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox


class CsvViewer:

    def __init__(self):

        self.root = None  # ルートウィンドウ
        self.data = None  # 読み込まれたcsvファイルの内容
        self.tree = None  # csvファイルの内容を表示するttk.Treeviewテーブル

    def start(self):

        self.call_root_window()
        self.call_csv_reader_widget()
        self.call_treeview_widget()
        self.root.mainloop()

    def call_root_window(self):
        """
        ルートウィンドウを呼び出す
        """
        self.root = tk.Tk()
        self.root.geometry('500x500')
        self.root.title('CsvViewer')

    def call_csv_reader_widget(self):
        """
        csvファイルを読み込むためのウィジェットを呼び出す関数
        """
        # widget配置のフレームを作成
        frame = tk.Frame(self.root, relief="ridge", bd=1)
        frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # ラベルを作成
        tk.Label(frame, text='Reference file >>').pack(side=tk.LEFT)

        # CSVファイルのファイルパスを指定する入力フィールドを作成
        entry_field = tk.Entry(frame)
        entry_field.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # ファイルダイアログを呼び出すボタンを作成
        tk.Button(frame, text='...', command=lambda: self.set_path(entry_field)).pack(side=tk.LEFT)

        # CSVファイルを読み込み，Treeviewに内容を表示するボタンを作成
        tk.Button(frame, text='read',
                  command=lambda: self.read_csv(entry_field.get(),  # entry_fieldに入力されているファイルパス
                                                )).pack(side=tk.LEFT)

    def call_treeview_widget(self):
        """
        ttk.Treeviewを呼び出し，X軸Y軸のスクロールバーを追加する関数
        """
        # widget配置のフレームを作成
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # CSVファイルの内容を表示するTreeviewを作成
        self.tree = ttk.Treeview(frame)
        self.tree.column('#0', width=50, stretch=tk.NO, anchor=tk.E)
        self.tree.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)

        # X軸スクロールバーを追加する
        hscrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(xscrollcommand=lambda f, l: hscrollbar.set(f, l))
        hscrollbar.grid(row=1, column=0, sticky=tk.W + tk.E + tk.N + tk.S)

        # Y軸スクロールバーを追加する
        vscrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vscrollbar.set)
        vscrollbar.grid(row=0, column=1, sticky=tk.W + tk.E + tk.N + tk.S)

    def set_path(self, entry_field):
        """
        tk.Entryの内容をクリアした後にファイルダイアログを呼び出し
        選択したファイルパスを，tk.Entryに記入する関数。

        :param entry_field: tk.Entry
        """
        # tk.Entryに記入されている内容をクリアする
        entry_field.delete(0, tk.END)

        # 実行ファイルの絶対パスを取得する
        abs_path = os.path.abspath(os.path.dirname(__file__))

        # 初期ディレクトリを実行ファイルの絶対パスにしたファイルダイアログを呼び出す
        file_path = filedialog.askopenfilename(initialdir=abs_path)

        # ファイルダイアログの選択結果をtk.Entryの内容に挿入する
        entry_field.insert(tk.END, str(file_path))

    def read_csv(self, path):
        """
        ファイルパスの拡張子がcsvの場合に内容を読み込み，ttk.Treeviewに内容を表示する。
        拡張子がcsv以外の場合はメッセージボックスを表示する。

        :param path: tk.Entryに入力されている文字列
        """
        # ファイルパスから拡張子を取得する
        extension = os.path.splitext(path)[1]

        # 拡張子がcsvの場合
        if extension == '.csv':

            # csvファイルの内容を読み込む
            with open(path) as f:
                reader = csv.reader(f)
                self.data = [row for row in reader]

            self.show_csv()

        # 拡張子がcsv以外の場合
        else:
            messagebox.showwarning('warning', 'Please select a csv file.')

    def show_csv(self):
        """
        読み込んだcsvファイルの内容をttk.Treeviewに表示する。
        """
        # Treeviewの内容をクリアする
        self.tree.delete(*self.tree.get_children())

        # 列番号をTreeviewに追加する
        self.tree['column'] = np.arange(np.array(self.data).shape[1]).tolist()

        # 列のヘッダーを行進する
        for i in self.tree['column']:
            self.tree.column(i, width=100, anchor=tk.E)
            self.tree.heading(i, text=str(i))

        # 行番号及びCSVの内容を表示する
        for i, row in enumerate(self.data, 0):
            self.tree.insert('', 'end', text=i, values=row)


if __name__ == '__main__':
    viewer = CsvViewer()
    viewer.start()
