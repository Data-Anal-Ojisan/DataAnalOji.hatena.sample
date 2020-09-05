# -*- coding: utf-8 -*-
"""
@author: data-anal-ojisan
"""
import numpy as np
import pandas as pd

def plotSSM(mcmc_sample, time_vec, obs_vec, state_name, 
            graph_title, y_label, date_lavels):

    # 状態空間モデルを図示する関数
    #
    # Args:
    #   mcmc_sample : MCMCサンプル
    #   time_vec    : 時間軸(POSIXct)のベクトル
    #   obs_vec     : (必要なら)観測値のベクトル
    #   state_name  : 図示する状態の変数名
    #   graph_title : グラフタイトル
    #   y_label     : y軸のラベル
    #   date_labels : 日付の書式
    #
    # Returns:
    #   生成されたグラフ
    
    ## すべての時点の状態の、95%区間と中央値
    result_df = pd.DataFrame(np.zeros([mcmc_sample[state_name].shape[1], 3]))
    for i in range(mcmc_sample[state_name].shape[1]):
        result_df.iloc[i,:] = np.percentile(mcmc_sample[state_name][:,i], q=[2.5, 50, 97.5])
        
    # 列名の変更
    result_df.columns = ["lwr", "fit", "upr"]
    
    # 時間軸の追加
    result_df['time'] = time_vec
    
    # 観測値の追加
    if obs_vec.isnull == False:
        result_df['obs'] = obs_vec
        
    # 図示
    
    
    

