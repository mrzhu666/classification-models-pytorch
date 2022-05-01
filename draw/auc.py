import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import csv
import sys
import pandas as pd
import numpy as np
import seaborn as sns



def ro_curve(y_pred, y_label):
    '''
        y_pred is a list of length n.  (0,1)    
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
    '''
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    fpr = dict()
    tpr = dict() 
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])  # 计算ROC曲线的auc值
    
    # plt.plot(fpr[0], tpr[0],
        #  lw=lw, label= method_name + ' (area = %0.2f)' % roc_auc[0])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    return fpr[0],tpr[0],roc_auc[0]

def GAN():
    # 标签列，以及对应的值，不要求是概率
    # lable:0,1
    file='GAN_eval.csv'
    f1 = pd.read_csv("result/"+file)
    y_label=f1.loc[:,"label"].to_numpy()
    y_pred=f1.loc[:,"prob"].to_numpy()
    return y_pred,y_label

def NO_GAN():
    # 标签列，以及对应的值，不要求是概率
    # lable:0,1
    file='No_GAN_eval.csv'
    f1 = pd.read_csv("result/"+file)
    y_label=f1.loc[:,"label"].to_numpy()
    y_pred=f1.loc[:,"prob"].to_numpy()
    return y_pred,y_label

def col_pic():
    fontsize = 14
    lw = 3
    

    y_pred,y_label=NO_GAN()
    fpr,tpr,roc_auc=ro_curve(y_pred,y_label)
    sns.lineplot(fpr,tpr,label= 'No GAN' + ' (area = %0.2f)' % roc_auc,lw=lw)  # 折线图

    y_pred,y_label=GAN()
    fpr,tpr,roc_auc=ro_curve(y_pred,y_label)
    sns.lineplot(fpr,tpr,label= 'GAN' + ' (area = %0.2f)' % roc_auc,lw=lw)  # 折线图




    sns.lineplot([0, 1], [0, 1], color='navy', linestyle='--')  # 对角线

    plt.title("ROC and AUC",fontsize=fontsize)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    # plt.xticks(font="Times New Roman",size=18,wei ght="bold")
    # plt.yticks(font="Times New Roman",size=18,weight="bold")
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    plt.legend(loc="lower right")
    plt.savefig("result/ROC_Comparison_GAN" + ".png",dpi=700)
    

def main():
    col_pic()
    
if __name__=="__main__":
    main() 