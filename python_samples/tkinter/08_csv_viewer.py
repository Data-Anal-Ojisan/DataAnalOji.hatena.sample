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

        self.root = None
        self.data = None
        self.tree = None

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
        csvファイルを読み込むためのウィジェットを呼び出す
        """
        # widgets frame
        frame = tk.Frame(self.root, relief="ridge", bd=1)
        frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # label
        tk.Label(frame, text='Reference file >>').pack(side=tk.LEFT)

        # entry field to define file path will be read
        entry_field = tk.Entry(frame)
        entry_field.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # button to call filedialog
        tk.Button(frame, text='...', command=lambda: self.set_path(entry_field)).pack(side=tk.LEFT)

        # button to load csv/xlsx
        tk.Button(frame, text='read',
                  command=lambda: self.read_csv_excel(entry_field.get(),  # file path from entry field
                                                      )).pack(side=tk.LEFT)

    def call_treeview_widget(self):
        """
        ttk.Treeviewを呼び出し，X軸Y軸のスクロールバーを追加する。
        """

        # ttk.Treeviewの呼び出し
        frame = tk.Frame(self.root);
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree = ttk.Treeview(frame)



        # X軸スクロールバーの追加
        hscrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(xscrollcommand=lambda f, l: hscrollbar.set(f, l))


        #　Y軸スクロールバーの追加
        vscrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vscrollbar.set)



        self.tree.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        hscrollbar.grid(row=1, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        vscrollbar.grid(row=0, column=1, sticky=tk.W + tk.E + tk.N + tk.S)

    def show_csv(self):
        """

        :return:
        """
        # 列番号をTreeviewに追加する
        self.tree['column'] = np.arange(np.array(self.data).shape[1]).tolist()
        for i in self.tree['column']:
            self.tree.column(i, width=50)
            self.tree.heading(i, text=str(i))

        # 行番号及びCSVの内容を描画する
        for i, row in enumerate(self.data, 0):
            print(row)
            self.tree.insert('', 'end', text=i, values=row)

    def set_path(self, entry_field):
        """

        :param entry_field:
        :return:
        """
        # clear entry_field
        entry_field.delete(0, tk.END)
        # get abspath
        abs_path = os.path.abspath(os.path.dirname(__file__))
        # call file dialog
        file_path = filedialog.askopenfilename(initialdir=abs_path)
        # set file path to entry_field
        entry_field.insert(tk.END, str(file_path))

    def read_csv_excel(self, path):
        """
        read csv or excel file then return dataframe of its file.

        Parameters
        ----------
        :param path: read csv or excel file then return dataframe of its file
        """

        # get a type of extension from file path
        extension = os.path.splitext(path)[1]

        # read dataset according to the extension
        if extension == '.csv':

            # read csv data
            with open(path) as f:
                reader = csv.reader(f)
                self.data = [row for row in reader]

            self.show_csv()

        else:
            messagebox.showwarning('warning', 'Please select a csv file.')


if __name__ == '__main__':
    viewer = CsvViewer()
    viewer.start()