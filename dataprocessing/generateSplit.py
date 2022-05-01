
from time import sleep, time
import cv2
import os
import sys
# sys.path.append("../")
sys.path.append("./")
print(sys.path)
from tqdm import tqdm
from shutil import copyfile
from setting import config
import shutil

# 毕设数据集预处理
# 训练集每一类拿取N张图片
# 验证集，全部tif转换为png复制到val文件夹
# sleep 避免电脑崩溃


data_path=config['server_path']
gan='CRC-HE'
# 数据转移整理后,server_path下有一个名为gan文件夹，里面有train和val文件夹
# 从原数据中


if os.path.exists(os.path.join(data_path,gan)):
    shutil.rmtree(os.path.join(data_path,gan))
    os.makedirs(os.path.join(data_path,gan))

# 训练集中每一类图片抽取的图片数量
N_TRAIN=1000
# 测试集中每一类图片抽取的图片数量
N_TEST=250


# NCT_path=data_path+'NCT-CRC-HE-100K-PNG/'
NCT_path=os.path.join(data_path,'NCT-CRC-HE-100K-PNG')


folders=os.listdir(NCT_path)  # 目录下的各种文件夹


data_target_path=os.path.join(data_path,gan)
NCT_target_path=os.path.join(data_target_path,'train')

print(NCT_path)

for fold in tqdm(folders):
    imgs=os.listdir(os.path.join(NCT_path,fold))
    if not os.path.exists(os.path.join(NCT_target_path,fold)):
        os.makedirs(os.path.join(NCT_target_path,fold))
    # sleep(1)
    for i,img in tqdm(enumerate(imgs)):
        if i==N_TRAIN:
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
    # sleep(1)
    for i,img in tqdm(enumerate(imgs)):
        if i==N_TEST:
            break
        tif=cv2.imread(os.path.join(NCT_path,fold,img))
        pngFile=img.replace('.tif','.png')
        cv2.imwrite(os.path.join(NCT_target_path,fold,pngFile),tif)