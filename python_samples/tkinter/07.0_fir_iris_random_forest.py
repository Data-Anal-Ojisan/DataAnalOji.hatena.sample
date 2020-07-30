# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:43:48 2020

@author: data-anal-ojisan
"""

import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# load dataset
iris = load_iris()
feature = iris.data
target = iris.target.reshape(-1,1)

# define model
RandomForestIris = RandomForestClassifier()

# fit
RandomForestIris.fit(feature, target)

# save model
with open('RandomForest_Iris.pickle', mode='wb') as fp:
    pickle.dump(RandomForestIris, fp)