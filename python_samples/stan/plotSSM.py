# -*- coding: utf-8 -*-
"""
@author: data-anal-ojisan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["font.family"] = "Meiryo"

def plotSSM(mcmc_sample, time_vec, obs_vec, state_name, 
            graph_title, y_label):

    # 状態空間モデルを図示する関数
    #
    # Args:
    #   mcmc_sample : MCMCサンプル
    #   time_vec    : 時間軸(POSIXct)のベクトル
    #   obs_vec     : (必要なら)観測値のベクトル
    #   state_name  : 図示する状態の変数名
    #   graph_title : グラフタイトル
    #   y_label     : y軸のラベル
    #
    # Returns:
    #   生成されたグラフ
    
    # すべての時点の状態の、95%区間と中央値
    result_df = pd.DataFrame(np.zeros([mcmc_sample[state_name].shape[1], 3]))
    for i in range(mcmc_sample[state_name].shape[1]):
        result_df.iloc[i,:] = np.percentile(mcmc_sample[state_name][:,i], q=[2.5, 50, 97.5])
        
    # 列名の変更
    result_df.columns = ["lwr", "fit", "upr"]
    
    # 時間軸の追加
    result_df['time'] = time_vec
    
    # 観測値の追加
    if obs_vec.isnull().all(axis=0) == False:
        result_df['obs'] = obs_vec
        
    # 図示
    plt.figure(figsize = (15,5))
    plt.plot(result_df['time'], 
             result_df['fit'], 
             color='black')
    plt.fill_between(x=result_df['time'],
                     y1=result_df['upr'],
                     y2=result_df['lwr'],
                     color='gray',
                     alpha=0.5)
    plt.ylabel(y_label)
    plt.title(graph_title)
    
    # 観測値をグラフに追加
    if obs_vec.isnull().all(axis=0) == False:
        plt.plot(result_df['time'],
                 result_df['obs'],
                 marker='.',
                 linewidth=0,
                 color='black')
        
    # グラフを返す
    plt.show()