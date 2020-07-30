# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 23:36:06 2020

@author: data-anal-ojisan
"""

# coding: utf-8
# cx_Freeze 用セットアップファイル
 
import sys
from cx_Freeze import setup, Executable
 
base = None

# GUI=有効, CUI=無効 にする
if sys.platform == 'win32' : base = 'Win32GUI'
 
# exe にしたい python ファイルを指定
exe = Executable(script = '07.1_iris_classifer.py',
                 base = base)
 
# セットアップ
setup(name = 'hello',
      version = '0.1',
      description = 'converter',
      executables = [exe])