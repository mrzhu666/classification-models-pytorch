# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import imp
import os
import sys
import argparse
import shutil
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from setting import config
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

if torch.cuda.is_available():
    torch.cuda.set_device(0)

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():    
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()
    print('eval_train')
    test_loss = 0.0 # cost function error
    correct = 0.0

    for batch_index, (images, labels) in enumerate(test_loader):

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        print('Test Epoch: {epoch} [{trained_samples}/{total_samples}]\t'.format(
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(test_loader.dataset)
        ))

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    print()
    

    #add informations to tensorboard

    writer.add_scalar('Test/Aver Age Loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)

# CRC-HE???CRC-HE-TRAIN ?????????CRC-HE-GAN?????????
def folder_fusion():    
    CRC_HE=os.path.join(config['server_path'],'CRC-HE')  # ???????????????
    CRC_HE_TRAIN=os.path.join(config['server_path'],'CRC-HE-TRAIN')  # GAN???????????????
    CRC_HE_GAN=os.path.join(config['server_path'],'CRC-HE-GAN')
    if os.path.exists(CRC_HE_GAN):
        shutil.rmtree(CRC_HE_GAN)

    shutil.copytree(CRC_HE,CRC_HE_GAN)

    # CRC_HE_TRAIN?????????CRC_HE_GAN
    folders=os.listdir(CRC_HE_TRAIN)
    for folder in folders:
        
        files=os.listdir(os.path.join(CRC_HE_TRAIN,folder))
        for file in files:
            shutil.copy(os.path.join(CRC_HE_TRAIN,folder,file),os.path.join(CRC_HE_GAN,'train',folder,file))
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')  # 128
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-gan_test', action='store_true', default=False, help='gan???????????????')

    args = parser.parse_args()
    args.gpu=torch.cuda.is_available()
    # args.b=256
    # ????????????
    num_workers=4

    net = get_network(args)

    if args.gan_test:
        folder_fusion()

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

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(config['checkpoint_path'], args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(config['checkpoint_path'], args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(config['checkpoint_path'], args.net, settings.TIME_NOW)


    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{acc}.pth')

    best_epoch = 0
    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(config['checkpoint_path'], args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(config['checkpoint_path'], args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(config['checkpoint_path'], args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(config['checkpoint_path'], args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(config['checkpoint_path'], args.net, recent_folder))


    for epoch in tqdm(range(1, settings.EPOCH + 1)):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        # if epoch > settings.MILESTONES[1] and best_acc < acc:
        if best_acc < acc:
            # weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            # print('saving weights file to {}'.format(weights_path))
            # torch.save(net.state_dict(), weights_path)
            best_acc = acc
            best_epoch = epoch
            # continue
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, acc=acc)
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

        # if not epoch % settings.SAVE_EPOCH:
        #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
        #     print('saving weights file to {}'.format(weights_path))
        #     torch.save(net.state_dict(), weights_path)

        print("Best acc:{:.4f}, Best epoch:{:d}".format(\
            best_acc,
            best_epoch
            ))
    writer.close()
