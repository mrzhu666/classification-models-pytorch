import os
import sys
from tqdm import tqdm
from shutil import copyfile

# 将原本的数据合并到生成的数据

origin='/home/student/mrzhu/data/CRC-HE/train/'
target='/home/student/mrzhu/data/CRC-HE-GAN/train/'

folders=os.listdir(origin)

for fold in tqdm(folders):
    imgs=os.listdir(os.path.join(origin,fold))
    for img in imgs:
        copyfile(os.path.join(origin,fold,img),os.path.join(target,fold,img))