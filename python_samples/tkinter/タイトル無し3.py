import time
import threading
import tkinter as tk

class GUI(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.value = 0

    def start(self):
        self.root = tk.Tk()
        # StringVarをフィールドに定義する
        self.sv = tk.StringVar()
        self.sv.set("0")
        # ラベルの表示 データはStringVarをバインドする
        self.label = tk.Label(self.root, textvariable=self.sv)
        self.label.pack()
        # ボタンの表示
        self.button = tk.Button(self.root, text='push', command=self.change_value_callback)
        self.button.pack()
        self.root.mainloop()

    # change_valueを別スレッドで実行するコールバック
    def change_value_callback(self):
        th = threading.Thread(target=self.change_value, args=())
        th.start()

    # StringVarを更新するように変更する
    def change_value(self):

        for value in range(100):
            time.sleep(0.05)
            # StringVarを変更するとGUIスレッドでラベル文字列が更新される
            self.sv.set(str(value))
            # ラベルに表示されるだろう値を表示
            print(value)

if __name__ == '__main__':
    gui = GUI()
    gui.start()