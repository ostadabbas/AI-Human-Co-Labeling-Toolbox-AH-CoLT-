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
from load_custom_dataset import load_rgbd_data
from models import *
from transform_newpad import NewPad
from utils import progress_bar
import cv2

import numpy as np
import matplotlib.pyplot as plt

import time

# Training
# change the config

lr = 0.00006
num_epoch = 9
train_batch_size = 128
test_batch_size = 199


train_losses_list = []
train_losses_class_list = []
train_acces_list = []

test_losses_list = []
test_losses_class_list = []
test_acces_list = []

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    '''
    num_supine = 0
    num_prone = 0
    num_sitting = 0
    num_standing = 0
    right_standing = 0
    right_supine = 0
    right_prone = 0
    right_sitting = 0
    '''

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, _, h, w = inputs.size()
        print(np.shape(inputs))
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_losses_class_list.append(loss)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print(targets)  # tensor
        '''
        for i in range(len(targets)):
            if targets[i] == 0:
                num_supine += 1
                if predicted[i] == targets[i]:
                    right_supine += 1
            if targets[i] == 1:
                num_prone += 1
                if predicted[i] == targets[i]:
                    right_prone += 1
            if targets[i] == 2:
                num_sitting += 1
                if predicted[i] == targets[i]:
                    right_sitting += 1
            if targets[i] == 3:
                num_standing += 1
                if predicted[i] == targets[i]:
                    right_standing += 1
        '''

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        train_losses_list.append(train_loss / (batch_idx + 1))
        # print(losses_list)
        train_acces_list.append(100. * correct / total)
        # print(acces_list)
    '''
    print('supine: ', right_supine/num_supine)
    print('/n')
    print('prone: ', right_prone/num_prone)
    print('/n')
    print('sitting: ', right_sitting/num_sitting)
    print('/n')
    print('standing: ', right_standing/num_standing)
    '''
    # print(num_supine, num_prone, num_sitting, num_standing)

def test(epoch,dir_folder):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    num_supine = 0
    num_prone = 0
    num_sitting = 0
    num_standing = 0
    right_standing = 0
    right_supine = 0
    right_prone = 0
    right_sitting = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_losses_class_list.append(loss)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for i in range(len(targets)):
                if targets[i] == 0:
                    num_supine += 1
                    if predicted[i] == targets[i]:
                        right_supine += 1
                if targets[i] == 1:
                    num_prone += 1
                    if predicted[i] == targets[i]:
                        right_prone += 1
                if targets[i] == 2:
                    num_sitting += 1
                    if predicted[i] == targets[i]:
                        right_sitting += 1
                if targets[i] == 3:
                    num_standing += 1
                    if predicted[i] == targets[i]:
                        right_standing += 1

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            test_losses_list.append(test_loss / (batch_idx + 1))
            # print(losses_list)
            test_acces_list.append(100. * correct / total)
            # print(acces_list)
        print('supine: ', right_supine/num_supine)
        print('/n')
        print('prone: ', right_prone/num_prone)
        print('/n')
        print('sitting: ', right_sitting/num_sitting)
        print('/n')
        print('standing: ', right_standing/num_standing)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir('Classifier/checkpoint_rgbd'):
        #     os.mkdir('Classifier/checkpoint_rgbd')
        torch.save(state, os.path.join(dir_folder, 'rgbd_ckpt.pth'))
        best_acc = acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RGBD-based Posture Classifier Training')
    parser.add_argument('--train_imgs',
                        default='/home/faye/Documents/FiDIP_Posture/output/mimm/pose_resnet_50/384x288_d256x3_adam_lr1e-3_MIMM/results_1/rgbd_pose_train/',
                        type=str, help='train set')
    parser.add_argument('--test_imgs',
                        default='/home/faye/Documents/FiDIP_Posture/output/mimm/pose_resnet_50/384x288_d256x3_adam_lr1e-3_MIMM/results_1/rgbd_pose_validate/',
                        type=str, help='test set')
    parser.add_argument('--train_anno',
                        default='/home/faye/Documents/FiDIP_Posture/data/mimm/annotations/MIMM_keypoints_train_mimm.json',
                        type=str, help='train annotation')
    parser.add_argument('--test_anno',
                        default='/home/faye/Documents/FiDIP_Posture/data/mimm/annotations/MIMM_keypoints_validate_mimm.json',
                        type=str, help='test annotation')
    parser.add_argument('--dir', type=str, help='output folder')
    parser.add_argument('--lr', default=lr, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Data
    print('==> Preparing data..')

    transform = transforms.Compose([transforms.ToTensor()])
    #transform = transforms.Compose([transforms.ToTensor(),
     #                               transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])

    trainset = load_rgbd_data(args.train_imgs, args.train_anno, transforms = transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,  batch_size=train_batch_size, shuffle=True, num_workers=1)

    testset = load_rgbd_data(args.test_imgs, args.test_anno, transforms = transform)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size= test_batch_size, shuffle=False, num_workers=1)

    classes = ('Supine', 'Prone', 'Sitting', 'Standing')

    # Model
    print('==> Building model..')
    net = Classifier_rgbd()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                       momentum=0.9, weight_decay=5e-4)

    nowTime = time.strftime('%Y%m%d%H%M')
    args.dir = 'rgbd_output_' + nowTime
    os.mkdir(args.dir)
    for epoch in range(start_epoch, start_epoch + num_epoch):
        train(epoch)
        print("test...")
        test(epoch, args.dir)
    

    np.save(os.path.join(args.dir,'train_losses_class.npy'), np.array(train_losses_class_list))
    np.save(os.path.join(args.dir,'train_losses.npy'), np.array(train_losses_list))
    np.save(os.path.join(args.dir,'train_accuracy.npy'), np.array(train_acces_list))

    np.save(os.path.join(args.dir, 'test_losses_class.npy'), np.array(test_losses_class_list))
    np.save(os.path.join(args.dir, 'test_losses.npy'), np.array(test_losses_list))
    np.save(os.path.join(args.dir, 'test_accuracy.npy'), np.array(test_acces_list))

    plt.subplot(311)
    plt.plot(train_losses_list)
    #plt.title('Loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')

    plt.subplot(312)
    plt.plot(train_acces_list)
    #plt.title('Accuracy')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')

    plt.subplot(313)
    plt.plot(train_losses_class_list)
    #plt.title('Class_Loss')
    plt.xlabel('iterations')
    plt.ylabel('class_loss')

    file1 = os.path.join(args.dir, 'train_Figure1.png')
    plt.savefig(file1)
    plt.show()

    plt.subplot(311)
    plt.plot(test_losses_list)
    #plt.title('Loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')

    plt.subplot(312)
    plt.plot(test_acces_list)
    #plt.title('Accuracy')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')

    plt.subplot(313)
    plt.plot(test_losses_class_list)
    #plt.title('Class_Loss')
    plt.xlabel('iterations')
    plt.ylabel('class_loss')

    file2 = os.path.join(args.dir, 'test_Figure1.png')
    plt.savefig(file2)
    plt.show()

