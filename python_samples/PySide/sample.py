#!/usr/bin/python
# -*- coding: utf-8 -*-

from sys import exit, argv
from PySide2.QtWidgets import QApplication, QLabel

def Main():
    app = QApplication(argv)
    label = QLabel("Hello World")
    label.show()
    exit(app.exec_())

if __name__ == '__main__':
    Main()