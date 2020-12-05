# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 23:36:06 2020

@author: data-anal-ojisan
"""
from distutils.core import setup
import py2exe
 
option = {
    'includes':['tkinter'],
    'compressed': 1,
    'optimize': 0,
    'bundle_files': 2,
}
 
setup(
    options = {
        'py2exe': option,
    },
    console = [
        {'script': '07.1_iris_classifier.py'}
    ],
    zipfile = None
)