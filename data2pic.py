#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 2018

@author: raine
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt 

def load_data_df(drivingcycle):
    vs_df = pd.read_csv(drivingcycle)
    vs_df = vs_df.head(5000)
    
    vs_df['I'] = vs_df['bat_I'] + vs_df['SC_I']
    vs_df['I.1'] = vs_df['bat_I.1'] + vs_df['SC_I.1']
    
    #ha = LA92_df[['I']]
    #ha['I_load'] = LA92_df[['I_Load']]
    #ha.plot()
    vs_df['v_dem'] = vs_df['EVPower']/vs_df['I_Load']
    vs_df['v_dem.1'] = vs_df['EVPower.1']/vs_df['I_Load.1']
    vs_df['v'] = vs_df['EVPower']/vs_df['I']
    vs_df['v.1'] = vs_df['EVPower.1']/vs_df['I.1']
    vs_df.fillna(150,inplace = True)
    #print(vs_df.shape[0] - vs_df.count())
#    print(vs_df.columns)
    return vs_df

def draw_7_8(df):
    t1 = np.arange(0, 500, 0.1)

    plt.figure(figsize = (15,25))
    
    no_pic = 10
    # 总线需要的电流 vs 实际电流
    plt.subplot(no_pic,1,1)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, -11, 8])
    plt.axis([0, 500, -4, 8])
    I = plt.plot(t1, df['I'].values, 'y.', t1, df['I_Load'].values, 'r--',t1, df['I_Load'].values - df['I'].values,'k--',linewidth=2)
    plt.legend(handles = I, labels = ['I', 'I_Load','err'], loc = 'best')

    # 超级电容的SOC：两种方法
    plt.subplot(no_pic,1,2)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, 74, 90])
    plt.axis([0, 500, 79, 87])
    SC_SOC = plt.plot(t1, df['SC_SOC'].values, 'y.', t1, df['SC_SOC.1'].values, 'r--',linewidth=2)
    plt.legend(handles = SC_SOC, labels = ['SC_SOC proposed', 'SC_SOC conventional'], loc = 'best')

    # 超级电容的电流：两种方法
    plt.subplot(no_pic,1,3)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, -11, 10])
    plt.axis([0, 500, -8, 10])
    SC_I = plt.plot(t1, df['SC_I'].values, 'y.', t1, df['SC_I.1'].values, 'r--',linewidth=2)
    plt.legend(handles = SC_I, labels = ['I_SC proposed', 'I_SC conventional'], loc = 'best')

    # 电池的电流：两种方法
    plt.subplot(no_pic,1,4)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, -9, 9])
    plt.axis([0, 500, -3, 6])
    bat_I = plt.plot(t1, df['bat_I'].values, 'g--', t1, df['bat_I.1'].values, 'r--',linewidth=2)
    plt.legend(handles = bat_I, labels = ['I_bat proposed', 'I_bat conventional'], loc = 'best')

    # 本文方法：电池的电流 vs 超级电容电流
    plt.subplot(no_pic,1,5)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, -9, 7])
    plt.axis([0, 500, -6, 5])
    I_proposed = plt.plot(t1, df['bat_I'].values, 'g--', t1, df['SC_I'].values, 'r--',t1, df['I_Load'].values, 'k--',linewidth=2)
    plt.legend(handles = I_proposed, labels = ['I_bat proposed', 'I_SC proposed'], loc = 'best')

    # 超级电容的电压：两种方法
    plt.subplot(no_pic,1,6)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, 15, 17.5])
    plt.axis([0, 500, 15, 17])
    V_SC = plt.plot(t1, df['SC_V'].values, 'g--', t1, df['SC_V.1'].values, 'r--',linewidth=2)
    plt.legend(handles = V_SC, labels = ['V_SC proposed', 'V_SC conventional'], loc = 'best')

    # 电池的SOC：两种方法
    plt.subplot(no_pic,1,7)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, 95.3, 95.8])
    plt.axis([0, 500, 95.65, 95.8])
    bat_SOC = plt.plot(t1, df['bat_SOC'].values, 'g--', t1, df['bat_SOC.1'].values, 'r--',linewidth=2)
    plt.legend(handles = bat_SOC, labels = ['bat_SOC proposed', 'bat_SOC conventional'], loc = 'best')

    plt.show()
    
def draw_6(NYCC_df,LA92_df):
    # 获取 LA92和 NYCC上的速度
    df = LA92_df[['Speed']]
    df['Speed.1'] = NYCC_df[['Speed']]
    # 获取 LA92和 NYCC上的功率
    df['EVPower'] = LA92_df[['EVPower']]*316.99999999999994/1000  #*缩放比例并换算为 kw
    df['EVPower.1'] = NYCC_df[['EVPower']]*316.99999999999994/1000
    df.index = LA92_df[['time']]
    df.columns = ['LA92  speed[km/h]','NYCC  speed[km/h]','LA92  power[kw]','NYCC  power[kw]']
    
    t1 = np.arange(0, 500, 0.1)

    plt.figure(figsize = (15,5))

    no_pic = 2
    # 总线需要的电流 vs 实际电流
    plt.subplot(no_pic,1,1)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, 0, 100])
    I = plt.plot(t1, df['LA92  speed[km/h]'].values, 'y.', t1, df['NYCC  speed[km/h]'].values, 'r--',linewidth=2)
    plt.legend(handles = I, labels = ['LA92  speed[km/h]', 'NYCC  speed[km/h]'], loc = 'best')

    # 超级电容的SOC：两种方法
    plt.subplot(no_pic,1,2)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, -70, 50])
    SC_SOC = plt.plot(t1, df['LA92  power[kw]'].values, 'y.-', t1, df['NYCC  power[kw]'].values, 'r--',linewidth=2)
    plt.legend(handles = SC_SOC, labels = ['LA92  power[kw]', 'NYCC  power[kw]'], loc = 'best')

    plt.show()

if __name__ == '__main__' :
    #获取路况的数据。功率放缩过，缩小了317倍
    NYCC_df = load_data_df("NYCC_reduce.csv")
    LA92_df = load_data_df("LA92_reduce.csv")
    # 产出图6
    draw_6(NYCC_df,LA92_df)
    # 产出图7
    draw_7_8(NYCC_df)
    # 产出图8
    draw_7_8(LA92_df)
    
    
    