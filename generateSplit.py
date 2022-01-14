import cv2
import os
import sys
from tqdm import tqdm
from shutil import copyfile
from setting import config

# 某一类图片抽取的图片数量

# 每一类拿取多少图片
N=1000
gan=''
data_path=config['server_path']+'CRC-HE/'
NCT_path=data_path+'NCT-CRC-HE-100K-PNG/'

folders=os.listdir(NCT_path)

data_target_path=os.path.join('/home/student/mrzhu/data/CRC-HE/',gan)
NCT_target_path=os.path.join(data_target_path,'train')


for fold in tqdm(folders):
    imgs=os.listdir(os.path.join(NCT_path,fold))
    if not os.path.exists(os.path.join(NCT_target_path,fold)):
        os.makedirs(os.path.join(NCT_target_path,fold))
    for i,img in enumerate(imgs):
        if i==N:
            break
        copyfile(os.path.join(NCT_path,fold,img),os.path.join(NCT_target_path,fold,img))

# 测试集
data_path=config['server_path']+'CRC-HE/'
NCT_path=data_path+'CRC-VAL-HE-7K/'

folders=os.listdir(NCT_path)

data_target_path=os.path.join('/home/student/mrzhu/data/CRC-HE/',gan)
NCT_target_path=os.path.join(data_target_path,'val')

for fold in tqdm(folders):
    imgs=os.listdir(os.path.join(NCT_path,fold))
    if not os.path.exists(os.path.join(NCT_target_path,fold)):
        os.makedirs(os.path.join(NCT_target_path,fold))
    for img in imgs:
        tif=cv2.imread(os.path.join(NCT_path,fold,img))
        pngFile=img.replace('.tif','.png')
        cv2.imwrite(os.path.join(NCT_target_path,fold,pngFile),tif)