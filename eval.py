# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import imp
import os
import sys
from setting import config

import argparse
import shutil
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.nn.modules.activation import Softmax

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings

from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

if torch.cuda.is_available():
    torch.cuda.set_device(0)

@torch.no_grad()
def result_save(file=''):
    # 类别数量的单位矩阵
    unit=np.identity(len(test_loader.dataset.classes),dtype=int)

    net.eval()
    print('eval_train')
    test_loss = 0.0 # cost function error
    correct = 0.0
    prob=np.empty(shape=(0))
    label=np.empty(shape=(0))

    # images.shape:torch.Size([64, 3, 224, 224])
    # labels.shape:torch.Size([64])
    for _, (images, labels) in enumerate(test_loader):
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)  # torch.Size([64, 100])
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)  # torch.Size([64])
        correct += preds.eq(labels).sum()
        
        output=outputs.cpu().numpy()[:,:9]
        prob=np.append(prob,output.ravel())
        for lab in labels.cpu().numpy():
            label=np.append(label,unit[lab])
        
    data=pd.DataFrame({'label':label,'prob':prob})
    data.to_csv('result/'+file,index=False)



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')  # 128
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-gan_test', action='store_true', default=False, help='gan数据集测试')
    args = parser.parse_args()
    args.gpu=torch.cuda.is_available()
    # args.b=256
    # 线程数量
    num_workers=4
    
    net = get_network(args)
    
    # CRC-HE/VAL 文件夹进行验证测试
    training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=num_workers,
        batch_size=args.b,
        shuffle=True,
        gan_test=args.gan_test
    )

    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=num_workers,
        batch_size=args.b,
        shuffle=True,
        gan_test=args.gan_test
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


    # 选择加载的模型
    recent_folder = 'Sunday_17_April_2022_20h_02m_21s'
    best_weights_file= 'resnet18-55-0.7835555672645569.pth'
    # file保存的csv文件名称
    file='No_GAN_eval.csv'

    weights_path = os.path.join(config['checkpoint_path'], args.net, recent_folder, best_weights_file)
    net.load_state_dict(torch.load(weights_path))

    result_save(file)

    # 选择加载的模型
    recent_folder = 'Sunday_17_April_2022_21h_15m_55s'
    best_weights_file= 'resnet18-63-0.8008888959884644.pth'
    # file保存的csv文件名称
    file='GAN_eval.csv'

    weights_path = os.path.join(config['checkpoint_path'], args.net, recent_folder, best_weights_file)
    net.load_state_dict(torch.load(weights_path))

    result_save(file)