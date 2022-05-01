#!/bin/bash

# python train.py -net vgg16
# python dataprocessing/generateSplit.py
python train.py -net resnet18


# tensorboard --logdir runs/resnet18 --port 6060

# 利用GAN生成的数据训练
# python train.py -net resnet18 -gan_test

# 进行评价
# python eval.py -net resnet18 
