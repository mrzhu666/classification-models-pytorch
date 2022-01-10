import cv2
import os
import easyocr
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split
from setting import config

# 将图片合并存储为array
# 要估计内存大小,图片太多不能直接加载到内存

folders=os.listdir(config['server_path']+config['dataset_path'])
classes = sorted(entry.name for entry in os.scandir(config['server_path']+config['dataset_path']) if entry.is_dir())
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

trainNum=0
testNum=0
for folder in tqdm(folders):
    files=os.listdir(os.path.join(config['server_path'],config['dataset_path'],folder))
    train, test = train_test_split(files, test_size=0.1, random_state = 42)
    trainNum+=len(train)
    testNum+=len(test)

indexTrain=0
trainData=np.empty(shape=(trainNum,3,224,224))
trainLabel=np.empty(shape=(0))
indexTest=0
testData=np.empty(shape=(testNum,3,224,224))
testLabel=np.empty(shape=(0))

for folder in tqdm(folders):
    files=os.listdir(os.path.join(config['server_path'],config['dataset_path'],folder))
    label=class_to_idx[folder]
    train, test = train_test_split(files, test_size=0.1, random_state = 42)
    
    for file in tqdm(train):
        img=cv2.imread(os.path.join(config['server_path'],config['dataset_path'],folder,file))  
        img=np.transpose(img,(2,0,1))[np.newaxis,:,:,:]  # (H,W,C) -> (1,C,H,W)
        # trainData=np.vstack((trainData,img))
        trainData[indexTrain]=img
        indexTrain+=1
    trainLabel=np.hstack((trainLabel,np.ones(shape=(len(train)))*label))

    for file in tqdm(test):
        img=cv2.imread(os.path.join(config['server_path'],config['dataset_path'],folder,file))  
        img=np.transpose(img,(2,0,1))[np.newaxis,:,:,:]  # (H,W,C) -> (1,C,H,W)
        # testData=np.vstack((testData,img))
        testData[indexTest]=img
        indexTest+=1
    testLabel=np.hstack((testLabel,np.ones(shape=(len(test)))*label))

print(trainData.shape,trainLabel.shape,testData.shape,testLabel.shape)

print('数据保存')
with open(config['server_path']+config['dataset_path']+'data.pkl','wb')  as f:
    pickle.dump((trainData,trainLabel,testData,testLabel),f)