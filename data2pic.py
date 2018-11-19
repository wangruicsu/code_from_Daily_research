#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 2018

@author: raine
"""
"""
|-----------------------------------------------|
|the proposed strategy                          |
|-----------------------------------------------|
| 符号           |  说明                         |
|-----------------------------------------------|
| time          |  时间间隔0.1s                  |
| bat_I         |  电池端的参考电流               |
| bat_SOC       |  电池实测 SOC                  |
| bat_v         |  电池实测电压                  |
| --------------|------------------------------|
| SC_I          |  超级电容端的参考电流            |
| SC_SOC        |  超级电容实测 SOC              |
| SC_v          |  超级电容实测电压               |
|---------------|----------------------------——|
| Speed         |  当前工况速度（放缩比例：10）    |
| EVPower       |  负载需求功率（放缩比例：13.2）  |
| Acceleration  |  当前工况加速度                |
| I_Load        |  总线实测电流                  |
|--------------------------------------------——|
|the conventional method：以.1为命名后缀         |
|----------------------------------------------|
"""
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt 

def load_data_df(drivingcycle):
    vs_df = pd.read_csv(drivingcycle)
    vs_df = vs_df.head(5000)
#    print(vs_df.columns)

    vs_df['v_bus'] = vs_df['EVPower']/vs_df['I_Load']
    vs_df['v_bus.1'] = vs_df['EVPower']/vs_df['I_Load.1']

    vs_df.fillna(24,inplace = True)
    #print(vs_df.shape[0] - vs_df.count())
#    print(vs_df.columns)
    return vs_df

def draw_7_8(df):
    t1 = np.arange(0, 500, 0.1)

    plt.figure(figsize = (15,25))
    
    no_pic = 10
    # 总线实际的电压 vs 总线参考电压 24v
    ax1 = plt.subplot(no_pic,1,1)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, 23.8, 24.2])
#    plt.axis([0, 500, 23.8, 24.2])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    v_bus_set = 24 * np.ones( (5000,1), dtype=np.int16 )
    v_bus = plt.plot(t1, df['v_bus'].values, 'y.', t1, df['v_bus.1'].values, 'r--',linewidth=2)
    plt.legend(handles = v_bus, labels = ['v_bus proposed', 'v_bus conventional'], loc = 'best',fontsize=15)
    ax1.set_ylabel('Voltage[v]', fontsize=20)

    # 超级电容的SOC：两种方法
    ax2 = plt.subplot(no_pic,1,2)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, 75, 95])
#    plt.axis([0, 500, 80, 89])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    X_1,Y_1 = add_guass(t1, df['SC_SOC'].values,sigma = 0.1)
    X_2,Y_2 = add_guass(t1, df['SC_SOC.1'].values,sigma = 0.1)
    SC_SOC = plt.plot(X_1,Y_1, 'y.', X_2,Y_2, 'r--',linewidth=2)
    plt.legend(handles = SC_SOC, labels = ['SC_SOC proposed', 'SC_SOC conventional'], loc = 'best',fontsize=15)
    ax2.set_ylabel('SOC[%]', fontsize=20)

    # 超级电容的电流：两种方法
    ax3 = plt.subplot(no_pic,1,3)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, -11, 10])
#    plt.axis([0, 500, -12, 14])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    X_1,Y_1 = add_guass(t1, df['SC_I'].values,sigma = 0.1)
    X_2,Y_2 = add_guass(t1, df['SC_I.1'].values,sigma = 0.1)
    SC_I = plt.plot(X_1,Y_1, 'y.', X_2,Y_2, 'r--',linewidth=2)
    plt.legend(handles = SC_I, labels = ['I_SC proposed', 'I_SC conventional'], loc = 'best',fontsize=15)
    ax3.set_ylabel('Current[A]', fontsize=20)

    # 电池的电流：两种方法
    ax4 = plt.subplot(no_pic,1,4)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, -11, 11])
#    plt.axis([0, 500, -5, 10])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    X_1,Y_1 = add_guass(t1, df['bat_I'].values,sigma = 0.1)
    X_2,Y_2 = add_guass(t1, df['bat_I.1'].values,sigma = 0.1)
    bat_I = plt.plot(X_1,Y_1, 'y--', X_2,Y_2, 'r--',linewidth=2)
    plt.legend(handles = bat_I, labels = ['I_bat proposed', 'I_bat conventional'], loc = 'best',fontsize=15)
    ax4.set_ylabel('Current[A]', fontsize=20)

    # 本文方法：电池的电流 vs 超级电容电流
    ax5 = plt.subplot(no_pic,1,5)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, -11, 9])
#    plt.axis([0, 500, -10, 10])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    X_1,Y_1 = add_guass(t1, df['bat_I'].values,sigma = 0.1)
    X_2,Y_2 = add_guass(t1, df['SC_I.1'].values,sigma = 0.1)
    I_proposed = plt.plot(X_1,Y_1, 'y--', X_2,Y_2, 'r--',linewidth=2)
    plt.legend(handles = I_proposed, labels = ['I_bat proposed', 'I_SC proposed'], loc = 'best',fontsize=15)
    ax5.set_ylabel('Current[A]', fontsize=20)

    # 超级电容的电压：两种方法
    ax6 = plt.subplot(no_pic,1,6)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, 14.5, 18])
#    plt.axis([0, 500, 15.5, 17.5])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    X_1,Y_1 = add_guass(t1, df['SC_V'].values,sigma = 0.05)
    X_2,Y_2 = add_guass(t1, df['SC_V.1'].values,sigma = 0.05)
    V_SC = plt.plot(X_1,Y_1, 'y--', X_2,Y_2, 'r--',linewidth=2)
    plt.legend(handles = V_SC, labels = ['V_SC proposed', 'V_SC conventional'], loc = 'best',fontsize=15)
    ax6.set_ylabel('Voltage[v]', fontsize=20)

    # 电池的SOC：两种方法
    ax7 = plt.subplot(no_pic,1,7)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, 95.0, 95.8])
#    plt.axis([0, 500, 95.6, 95.9])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    X_1,Y_1 = add_guass(t1, df['bat_SOC'].values,sigma = 0.01)
    X_2,Y_2 = add_guass(t1, df['bat_SOC.1'].values,sigma = 0.01)
    bat_SOC = plt.plot(X_1,Y_1, 'y--', X_2,Y_2, 'r--',linewidth=2)
    plt.legend(handles = bat_SOC, labels = ['bat_SOC proposed', 'bat_SOC conventional'], loc = 'best',fontsize=15)
    ax7.set_ylabel('SOC[%]', fontsize=20)

    plt.show()
    
def draw_6(NYCC_df,LA92_df):
    # 获取 LA92和 NYCC上的速度
    df = LA92_df[['Speed']]
    df['Speed.1'] = NYCC_df[['Speed']]
    
    # 获取 LA92和 NYCC上的功率
    df['EVPower'] = LA92_df[['EVPower']]*13.2/1000  #*缩放比例并换算为 kw
    df['EVPower.1'] = NYCC_df[['EVPower']]*13.2/1000
    df.index = LA92_df[['time']]
    df.columns = ['LA92  speed[km/h]','NYCC  speed[km/h]','LA92  power[kw]','NYCC  power[kw]']
    
    t1 = np.arange(0, 500, 0.1)

    plt.figure(figsize = (15,5))

    no_pic = 2
    # LA92和 NYCC 的 speed
    ax1 = plt.subplot(no_pic,1,1)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, 0, 120])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    I = plt.plot(t1, df['LA92  speed[km/h]'].values, 'y', t1, df['NYCC  speed[km/h]'].values, 'r--',linewidth=3)
#    ax1.set_xlabel('Time', fontsize=20)
    plt.legend(handles = I, labels = ['LA92', 'NYCC'], loc = 'best',fontsize=15)
    ax1.set_ylabel('Speed[Km/h]', fontsize=20)

    # LA92和 NYCC 的 power
    ax2 = plt.subplot(no_pic,1,2)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
    plt.axis([0, 500, -60, 40])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    SC_SOC = plt.plot(t1, df['LA92  power[kw]'].values, 'y', t1, df['NYCC  power[kw]'].values, 'r--',linewidth=3)
    plt.legend(handles = SC_SOC, labels = ['LA92', 'NYCC'], loc = 'best',fontsize=15)
    ax2.set_xlabel('Time[s]', fontsize=20)
    ax2.set_ylabel('Power[Kw]', fontsize=20)
    plt.show()
    
def add_guass(X_array,Y_array,sigma):
    # 对输入数据加入gauss噪声
    # 定义gauss噪声的均值和方差
    mu = 0
#    sigma = 0.1
    for i in range(X_array.size):
        X_array[i] += random.gauss(mu,sigma)
        Y_array[i] += random.gauss(mu,sigma)
    return X_array,Y_array

if __name__ == '__main__' :
    #获取路况的数据。功率放缩过，缩小了13.2倍
    NYCC_df = load_data_df("NYCC.csv")
    LA92_df = load_data_df("LA92.csv")
    # 产出图6
    draw_6(NYCC_df,LA92_df)
    # 产出图7
    draw_7_8(LA92_df)
    # 产出图8
    draw_7_8(NYCC_df)
    # todo
    # 可以用台式机跑哇？
    # 本次数据趋势不太 OK
    
    
    
    