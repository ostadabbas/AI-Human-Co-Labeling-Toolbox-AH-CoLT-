import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from load_custom_dataset import load_kpts3D_syrip_5class
from models import *
from transform_newpad import NewPad
from utils import progress_bar
import cv2

import numpy as np
import matplotlib.pyplot as plt

import time

# Training
# change the config

lr = 0.0008
num_epoch = 100
train_batch_size = 50
test_batch_size = 100


test_losses_list = []
test_losses_class_list = []
test_acces_list = []


def test(epoch, dir_folder):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    num_supine = 0
    num_prone = 0
    num_sitting = 0
    num_standing = 0
    num_allfours = 0
    num_ave = 0
    right_standing = 0
    right_supine = 0
    right_prone = 0
    right_sitting = 0
    right_allfours = 0
    right_ave = 0
    pred_list = []
    tar_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, imgs) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_losses_class_list.append(loss)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pred_list.append(predicted)
            tar_list.append(targets)
            print(predicted)
            print(targets)

            for i in range(len(targets)):
                if targets[i] == 0:
                    num_supine += 1
                    num_ave += 1
                    if predicted[i] == targets[i]:
                        right_supine += 1
                        right_ave += 1
                if targets[i] == 1:
                    num_prone += 1
                    num_ave += 1
                    if predicted[i] == targets[i]:
                        right_prone += 1
                        right_ave += 1
                if targets[i] == 2:
                    num_sitting += 1
                    num_ave += 1
                    if predicted[i] == targets[i]:
                        right_sitting += 1
                        right_ave += 1
                if targets[i] == 3:
                    num_standing += 1
                    num_ave += 1
                    if predicted[i] == targets[i]:
                        right_standing += 1
                        right_ave += 1
                if targets[i] == 4:
                    num_allfours += 1
                    num_ave += 1
                    if predicted[i] == targets[i]:
                        right_allfours += 1
                        right_ave += 1

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            test_losses_list.append(test_loss / (batch_idx + 1))
            # print(losses_list)
            test_acces_list.append(100. * correct / total)
            # print(acces_list)
        print('average: ', right_ave/num_ave)
        print('/n')
        print('supine: ', right_supine/num_supine)
        print('/n')
        print('prone: ', right_prone/num_prone)
        print('/n')
        print('sitting: ', right_sitting/num_sitting)
        print('/n')
        print('standing: ', right_standing/num_standing)
        print('/n')
        print('allfours: ', right_allfours/num_allfours)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='3D Keypoints-based Posture Classifier Training')
   

    parser.add_argument('--train_anno',
                        default='/home/faye/Documents/InfantProject/data/SyRIP/test100_train600_2d/annotations/train600/person_keypoints_train_infant_530_5class.json',
                        type=str, help='train annotation')

    parser.add_argument('--test_anno',
                        default='/home/faye/Documents/InfantProject/data/SyRIP/test100_train600_2d/annotations/validate100/person_keypoints_validate_infant_5class.json',
                        type=str, help='test annotation')

    parser.add_argument('--dir', type=str, help='output folder')
    parser.add_argument('--lr', default=lr, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    #train_kpt = '/home/faye/Documents/InfantProject/data/SyRIP/test100_train600_3d/train530/correct_3D_530.npy'
    #val_kpt = '/home/faye/Documents/InfantProject/data/SyRIP/test100_train600_3d/validate100/correct_3D_100.npy'
    train_kpt = '/home/faye/Documents/InfantProject/data/SyRIP/test100_train600_3d/train530/output_pose_3D_530.npy'
    val_kpt = '/home/faye/Documents/InfantProject/data/SyRIP/test100_train600_3d/validate100/output_pose_3D_100.npy'

    name_list_train = '/home/faye/Documents/InfantProject/data/SyRIP/test100_train600_3d/train530/output_imgnames_530.npy'
    name_list_val = '/home/faye/Documents/InfantProject/data/SyRIP/test100_train600_3d/validate100/output_imgnames_100.npy'
    
    # Data  # 2 parameters to control the data load. kpt_file and train or not (using .npy)
    print('==> Preparing data..')
    trainset = load_kpts3D_syrip_5class(args.train_anno, train_kpt, name_list_train, transforms=None)

    trainloader = torch.utils.data.DataLoader(
        trainset,  batch_size=train_batch_size, shuffle=True, num_workers=1)

    testset = load_kpts3D_syrip_5class(args.test_anno, val_kpt, name_list_val, transforms = None)
    #testset = load_kpts_gt(args.test_anno, transforms = None)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size= test_batch_size, shuffle=False, num_workers=1)

    classes = ('Supine', 'Prone', 'Sitting', 'Standing')

    # Model
    print('==> Building model..')
    net = Classifier_kpts3d_5class()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    checkpoint = torch.load('/home/faye/Documents/FiDIP_Posture/Classifier/kpts_output_202208070628/kpts_ckpt.pth') 
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                       momentum=0.9, weight_decay=5e-4)
    nowTime = time.strftime('%Y%m%d%H%M')
    args.dir = 'val_kpts3d_output_' + nowTime
    if not os.path.exists(args.dir):
        os.mkdir(args.dir)
    print(args.dir)
    epoch = 1
    print("test...")
    test(start_epoch, args.dir)


    print(args.dir)

