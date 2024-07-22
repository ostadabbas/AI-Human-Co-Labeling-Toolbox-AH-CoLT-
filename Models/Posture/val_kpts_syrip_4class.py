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
from load_custom_dataset import load_kpts
from load_custom_dataset import load_kpts_gt
from models import *
from transform_newpad import NewPad
from utils import progress_bar
import cv2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.metrics import confusion_matrix
import seaborn as sn

# Training
# change the config

lr = 0.002
num_epoch = 1
train_batch_size = 139
test_batch_size = 1050



train_losses_list = []
train_losses_class_list = []
train_acces_list = []

test_losses_list = []
test_losses_class_list = []
test_acces_list = []

def plotCM(array):
    df_cm = pd.DataFrame(array, ["Supine", "Prone", "Sitting", "Standing"], ["Supine", "Prone", "Sitting", "Standing"])
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    plt.show()


def test(epoch, dir_folder):
    global best_acc
    # net = Classifier_kpts()
    net.eval()
    # net.load_state_dict('./Classifier/kpts_output/kpts_ckpt.pth')
    test_loss = 0
    correct = 0
    total = 0

    num_supine = 0
    num_prone = 0
    num_sitting = 0
    num_standing = 0
    num_ave = 0
    right_standing = 0
    right_supine = 0
    right_prone = 0
    right_sitting = 0
    right_ave = 0
    pred_list = []
    tar_list = []
    img_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, imgs) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # print(outputs)
            np.save(os.path.join(dir_folder,'outputs_kpts.npy'), outputs.cpu())
            loss = criterion(outputs, targets)
            test_losses_class_list.append(loss)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pred_list.append(predicted)
            tar_list.append(targets)
            img_list.append(imgs)

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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            test_losses_list.append(test_loss / (batch_idx + 1))
            # print(losses_list)
            test_acces_list.append(100. * correct / total)
            # print(acces_list)
        print('average: ', right_ave/num_ave)
        print('/n')
        print('supine: ', right_supine/num_supine, num_supine)
        print('/n')
        print('prone: ', right_prone/num_prone, num_prone)
        print('/n')
        print('sitting: ', right_sitting/num_sitting, num_sitting)
        print('/n')
        print('standing: ', right_standing/num_standing, num_standing)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        best_acc = acc
    # print(img_list)
    # print(tar_list, np.shape(tar_list))
    new_tar = tar_list[0].tolist()
    new_pred = pred_list[0].tolist()
    print(new_tar)
    cm = confusion_matrix(new_tar, new_pred, [0, 1, 2, 3])
    print(cm)
    plotCM(cm)
    np.save(os.path.join(dir_folder, 'pred.npy'), pred_list)
    np.save(os.path.join(dir_folder, 'tar.npy'), tar_list)
    np.save(os.path.join(dir_folder, 'img.npy'), img_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Keypoints-based Posture Classifier Training')
    # MIMM
    '''
    parser.add_argument('--test_pred',
                        default='//home/faye/Documents/InfantProject/outputs/FiDIP_models/new_mimm/darkpose/hrnet_mimm_216_validate/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_custom/results/keypoints_validate_infant_results_1.json',
                        type=str, help='test preds')
    
    parser.add_argument('--test_anno',
                        default='/home/faye/Documents/InfantProject/data/MIMM_new/annotations/validate_216/person_keypoints_validate_infant_2.json',
                        type=str, help='test annotation')
    #parser.add_argument('--test_anno',
   #                     default='/home/faye/Documents/InfantProject/data/MIMM_2D/annotations/mimm/person_keypoints_validate_infant.json',
    #                    type=str, help='test annotation')

    '''
    # SyRIP
    parser.add_argument('--test_pred',
                        default='/home/faye/Documents/InfantProject/outputs/FiDIP_models/Journal Paper Results/DarkFd93.6/keypoints_validate_infant_results_0.json',
                        type=str, help='test preds')

    
    parser.add_argument('--test_anno',
                        default='/home/faye/Documents/InfantProject/data/SyRIP/test100_train600_2d/annotations/validate100/person_keypoints_validate_infant.json',
                        type=str, help='test annotation')
    
    parser.add_argument('--dir', type=str, help='output folder')
    parser.add_argument('--lr', default=lr, type=float, help='learning rate')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Data
    print('==> Preparing data..')
    testset = load_kpts(args.test_pred, args.test_anno, transforms = None)
    #testset = load_kpts_gt(args.test_anno, transforms = None)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size= test_batch_size, shuffle=False, num_workers=1)

    classes = ('Supine', 'Prone', 'Sitting', 'Standing')

    # Model
    print('==> Building model..')
    net = Classifier_kpts()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # if args.resume:
    #     # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #checkpoint = torch.load('/home/faye/Documents/FiDIP_Posture/Posture_Results/kpts_output_202103030007/kpts_ckpt.pth') # previous best 4 posture classes model
    checkpoint = torch.load('/home/faye/Documents/FiDIP_Posture/Classifier/train_syrip_withoutTrans_kpts_output_202207281246/kpts_ckpt.pth')  # new 4 posture classes model trained without transition
    #checkpoint = torch.load('/home/faye/Documents/FiDIP_Posture/Classifier/paper_results/kpts_output_2dmimmon2dmimm/kpts_ckpt.pth')
    #checkpoint = torch.load('/home/faye/Documents/FiDIP_Posture/Classifier/kpts_output/kpts_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                       momentum=0.9, weight_decay=5e-4)
    nowTime = time.strftime('%Y%m%d%H%M')
    args.dir = 'kpts_output_' + nowTime
    os.mkdir(args.dir)

    for epoch in range(start_epoch, start_epoch + num_epoch):
        print("test...")
        test(epoch, args.dir)

    # np.save(os.path.join(args.dir,'train_losses_class.npy'), np.array(train_losses_class_list))
    # np.save(os.path.join(args.dir,'train_losses.npy'), np.array(train_losses_list))
    # np.save(os.path.join(args.dir,'train_accuracy.npy'), np.array(train_acces_list))
    #
    # np.save(os.path.join(args.dir, 'test_losses_class.npy'), np.array(test_losses_class_list))
    # np.save(os.path.join(args.dir, 'test_losses.npy'), np.array(test_losses_list))
    # np.save(os.path.join(args.dir, 'test_accuracy.npy'), np.array(test_acces_list))
    #
    # plt.subplot(311)
    # plt.plot(train_losses_list)
    # #plt.title('Loss')
    # plt.xlabel('iterations')
    # plt.ylabel('loss')
    #
    # plt.subplot(312)
    # plt.plot(train_acces_list)
    # #plt.title('Accuracy')
    # plt.xlabel('iterations')
    # plt.ylabel('accuracy')
    #
    # plt.subplot(313)
    # plt.plot(train_losses_class_list)
    # #plt.title('Class_Loss')
    # plt.xlabel('iterations')
    # plt.ylabel('class_loss')
    #
    # file1 = os.path.join(args.dir, 'train_Figure1.png')
    # plt.savefig(file1)
    # plt.show()
    #
    # plt.subplot(311)
    # plt.plot(test_losses_list)
    # #plt.title('Loss')
    # plt.xlabel('iterations')
    # plt.ylabel('loss')
    #
    # plt.subplot(312)
    # plt.plot(test_acces_list)
    # #plt.title('Accuracy')
    # plt.xlabel('iterations')
    # plt.ylabel('accuracy')
    #
    # plt.subplot(313)
    # plt.plot(test_losses_class_list)
    # #plt.title('Class_Loss')
    # plt.xlabel('iterations')
    # plt.ylabel('class_loss')
    #
    # file2 = os.path.join(args.dir, 'test_Figure1.png')
    # plt.savefig(file2)
    # plt.show()

    print(args.dir)

