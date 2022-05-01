#test.py
#!/usr/bin/env python3
import os
import shutil
from setting import config

def folder_fusion():
    CRC_HE=os.path.join(config['server_path'],'CRC-HE')  # 原本的样本
    CRC_HE_TRAIN=os.path.join(config['server_path'],'CRC-HE-TRAIN')  # GAN生成的样本
    CRC_HE_GAN=os.path.join(config['server_path'],'CRC-HE-GAN')
    if os.path.exists(CRC_HE_GAN):
        shutil.rmtree(CRC_HE_GAN)

    shutil.copytree(CRC_HE,CRC_HE_GAN)

    # CRC_HE_TRAIN融合到CRC_HE_GAN
    folders=os.listdir(CRC_HE_TRAIN)
    for folder in folders:
        
        files=os.listdir(os.path.join(CRC_HE_TRAIN,folder))
        for file in files:
            shutil.copy(os.path.join(CRC_HE_TRAIN,folder,file),os.path.join(CRC_HE_GAN,'train',folder,file))

    
folder_fusion()