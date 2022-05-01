
from typing import Tuple
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import csv
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns





def get_data(folder,file,item):
    #加载日志数据
    ea=event_accumulator.EventAccumulator(os.path.join('runs/resnet18',folder,file)) 
    ea.Reload()
    # print(ea.scalars.Keys())
    val_psnr=ea.scalars.Items(item)
    # print(len(val_psnr))
    data=[(i.step,i.value) for i in val_psnr]  # 迭代数和相应的值

    return zip(*data)

def NO_GAN(item)->Tuple[np.array,np.array]:
    folder='Sunday_17_April_2022_20h_02m_21s'
    file='events.out.tfevents.1650196946.container-8fb311963c-58ab387f.2917.0'
    return get_data(folder,file,item)

def GAN(item)->Tuple[np.array,np.array]:
    folder='Sunday_17_April_2022_21h_15m_55s'
    file='events.out.tfevents.1650201363.container-8fb311963c-58ab387f.282978.0'
    return get_data(folder,file,item)



file_save='test Accuracy.png'
fontsize = 14
item='Test/Accuracy'

plt.figure(figsize=[9.6, 6],dpi=150)

x,y=NO_GAN(item)
sns.lineplot(x,y,label='No-GAN')  # 折线图
x,y=GAN(item)
sns.lineplot(x,y,label='GAN')  # 折线图

plt.xlabel(item, fontsize = fontsize)
plt.title("Accuracy comparison between No-GAN and GAN",fontsize=fontsize)
plt.legend(loc="lower right")
plt.savefig(os.path.join('result',file_save))



file_save='train loss.png'
fontsize = 14
item='Train/loss'

plt.figure(figsize=[9.6, 6],dpi=150)

x,y=NO_GAN(item)
sns.lineplot(x,y,label='No-GAN')  # 折线图
x,y=GAN(item)
sns.lineplot(x,y,label='GAN')  # 折线图

plt.xlabel(item, fontsize = fontsize)
plt.title("Loss comparison between No-GAN and GAN",fontsize=fontsize)
plt.legend(loc="lower right")
plt.savefig(os.path.join('result',file_save))

