#!/bin/bash

python train.py -net vgg16
python train.py -net resnet18

tensorboard --logdir runs/resnet18 --port 6060