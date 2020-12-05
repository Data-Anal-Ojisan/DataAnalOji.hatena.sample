# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:43:48 2020

@author: data-anal-ojisan
"""

import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# irisデータセットを読み込む
iris = load_iris()                   # インスタンスを生成
feature = iris.data                  # 特徴量を取得
target = iris.target.reshape(-1, 1)  # クラスを取得

# RandomForestClassifierのインスタンスを生成
RandomForestIris = RandomForestClassifier()

# 学習を実行
RandomForestIris.fit(feature, target)

# モデルを保存
with open('model/RandomForest_Iris.pickle', mode='wb') as fp:
    pickle.dump(RandomForestIris, fp)
