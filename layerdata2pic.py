#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:06:16 2018

@author: raine
"""

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
#    vs_df = vs_df.head(5000)
#    print(vs_df.columns)
    vs_df.fillna(24,inplace = True)
    #print(vs_df.shape[0] - vs_df.count())
#    print(vs_df.columns)
    return vs_df

def draw_7_8(df):
    no_pic = 10
    t1 = np.arange(0, 1024, 1)
    
    plt.figure(figsize = (20,50))
    
    # 总线实际的电压 vs 总线参考电压 24v
    ax = plt.subplot(no_pic,2,1)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, 23.8, 24.2])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    v_bus_set = 24 * np.ones( (5000,1), dtype=np.int16 )
    v_bus = plt.plot(t1, df['X'].values, 'r-',linewidth=2)
    plt.legend(handles = v_bus,labels = ['V0'],fontsize=15)
    ax.set_ylabel('X', fontsize=20)

    
    # 总线实际的电压 vs 总线参考电压 24v
    ax1 = plt.subplot(no_pic,2,2)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, 23.8, 24.2])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    v_bus = plt.plot(t1, df['L1'].values, 'y-', t1, df['D1'].values, 'b-',linewidth=2)
    plt.legend(handles = v_bus, labels = ['V1', 'W1'], loc = 'best',fontsize=15)
    ax1.set_ylabel('One layer', fontsize=20)

#    plt.title('(a)')
    
    # 超级电容的SOC：两种方法
    ax2 = plt.subplot(no_pic,2,3)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, 80, 89])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    SC_SOC = plt.plot(t1,df['L2'].values, 'y-', t1,df['D2'].values, 'b-',linewidth=2)
    plt.legend(handles = SC_SOC, labels = ['V2', 'W2'], loc = 'best',fontsize=15)
    ax2.set_ylabel('Two layer', fontsize=20)

    # 超级电容的电流：两种方法
    ax3 = plt.subplot(no_pic,2,4)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, -12, 14])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    SC_I = plt.plot(t1, df['L3'].values, 'y-', t1, df['D3'].values, 'b-',linewidth=2)
    plt.legend(handles = SC_I, labels = ['V3', 'W3'], loc = 'best',fontsize=15)
    ax3.set_ylabel('Three layer', fontsize=20)
    
    # 电池的电流：两种方法
    ax4 = plt.subplot(no_pic,2,5)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, -5, 10])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    bat_I = plt.plot(t1, df['L4'].values, 'y-', t1, df['D4'].values, 'b-',linewidth=2)
    plt.legend(handles = bat_I, labels = ['V4', 'W4'], loc = 'best',fontsize=15)
    ax4.set_ylabel('Four layer', fontsize=20)


    # 本文方法：电池的电流 vs 超级电容电流
    ax5 = plt.subplot(no_pic,2,6)
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#    plt.axis([0, 500, -10, 10])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    I_proposed = plt.plot(t1, df['L5'].values, 'y-', t1, df['D5'].values, 'b-',linewidth=2)
    plt.legend(handles = I_proposed, labels = ['V5', 'W5'], loc = 'best',fontsize=15)
    ax5.set_ylabel('Five layer', fontsize=20)
    ax5.set_xlabel('Time[s]', fontsize=20)

    plt.show()
   

if __name__ == '__main__' :
    #获取路况的数据。功率放缩过，缩小了13.2倍
    NYCC_df = load_data_df("5.csv")
    print(NYCC_df.shape)
    # 产出图6
    draw_7_8(NYCC_df)


    
    
    
    