
from time import sleep, time
import os
import sys
# sys.path.append("../")
sys.path.append("./")
print(sys.path)
from tqdm import tqdm
from shutil import copyfile
from setting import config
import shutil



# 生成文件夹CRC-HE-FID
# 用于FID值评价
# 使用的两个文件夹CRC-HE/TRAIN,CRC-HE—TRAIN


CRC_HE=os.path.join(config['dataset_path'],'train')
CRC_HE_TRAIN=os.path.join(config['server_path'],'CRC-HE-TRAIN')

# 要复制到的目标文件夹
target=os.path.join(config['server_path'],'CRC-HE-FID')


# CRC-HE/TRAIN 复制

target_CRC_HE=os.path.join(target,'CRC-HE')

if not os.path.exists(target_CRC_HE):
    os.makedirs(target_CRC_HE)

folders=os.listdir(CRC_HE)  # 目录下的各种文件夹类
for fold in folders:
    imgs=os.listdir(os.path.join(CRC_HE,fold))
    for img in tqdm(imgs):
        copyfile(os.path.join(CRC_HE,fold,img),os.path.join(target_CRC_HE,img))
 

# CRC-HE-TRAIN 

target_CRC_HE_TRAIN=os.path.join(target,'CRC-HE-TRAIN')

if not os.path.exists(target_CRC_HE_TRAIN):
    os.makedirs(target_CRC_HE_TRAIN)

folders=os.listdir(CRC_HE_TRAIN)  # 目录下的各种文件夹类
for fold in folders:
    imgs=os.listdir(os.path.join(CRC_HE_TRAIN,fold))
    for img in tqdm(imgs):
        copyfile(os.path.join(CRC_HE_TRAIN,fold,img),os.path.join(target_CRC_HE_TRAIN,img))