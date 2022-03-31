
import cv2
import os
import sys
# sys.path.append("../")
sys.path.append("./")
print(sys.path)
from tqdm import tqdm
from shutil import copyfile
from setting import config

# 毕设数据集预处理
# 训练集每一类拿取多少图片

gan='CRC-HE'
data_path=config['server_path']
# 数据转移整理后
# server_path下有一个CRC-HE文件夹，里面有train和val文件夹

# 训练集
N=float('inf') # 某一类图片抽取的图片数量
# NCT_path=data_path+'NCT-CRC-HE-100K-PNG/'
NCT_path=os.path.join(data_path,'NCT-CRC-HE-50K-JPG')


folders=os.listdir(NCT_path)  # 目录下的各种文件夹


data_target_path=os.path.join(data_path,gan)
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
NCT_path=os.path.join(data_path,'CRC-VAL-HE-7K')


folders=os.listdir(NCT_path)


data_target_path=os.path.join(data_path,gan)
NCT_target_path=os.path.join(data_target_path,'val')

for fold in tqdm(folders):
    imgs=os.listdir(os.path.join(NCT_path,fold))
    if not os.path.exists(os.path.join(NCT_target_path,fold)):
        os.makedirs(os.path.join(NCT_target_path,fold))
    for img in imgs:
        tif=cv2.imread(os.path.join(NCT_path,fold,img))
        pngFile=img.replace('.tif','.jpg')
        cv2.imwrite(os.path.join(NCT_target_path,fold,pngFile),tif)