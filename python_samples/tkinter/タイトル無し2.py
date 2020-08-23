import time
import threading
import tkinter as tk

class GUI(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.value = 0
    
    def start(self):
        self.root = tk.Tk()
        # ラベルの表示
        self.label = tk.Label(self.root, text=self.value)
        self.label.pack()
        # ボタンの表示
        self.button = tk.Button(self.root, text='push', command=self.change_value)
        self.button.pack()
        self.root.mainloop()
    
        
    def change_value(self):
        
        for value in range(100):
            time.sleep(0.05)
            self.value = value
            # ラベルの値を変更
            self.label['text'] = str(self.value)
            # ラベルに表示されるだろう値を表示
            print(self.value)
    
if __name__ == '__main__':
    gui = GUI()
    gui.start()