import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from .load_custom_dataset import load_pkl_kpts_test  # Assuming this function can be used to load and preprocess the skeleton data
from .models import Classifier_kpts
from .utils import progress_bar
import json
import time

# Configuration parameters
test_batch_size = 1
lr = 0.002

def test(model, device, testloader):
    model.eval()
    pred_list = []
    img_list = []
    score_list = []

    with torch.no_grad():
        for batch_idx, (inputs, imgs) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            score_list.append(outputs.cpu().numpy())
            pred_list.append(predicted.cpu().numpy())
            img_list.append(imgs)

            progress_bar(batch_idx, len(testloader), 'Processing')

    # Flatten lists
    pred_list = np.concatenate(pred_list, axis=0)
    img_list = np.concatenate(img_list, axis=0)
    return pred_list, img_list, score_list

def inference_posture(kpt_pred):
    parser = argparse.ArgumentParser(description='Keypoints-based Posture Classifier Inference')
    args = parser.parse_args()
    args.test_pred = kpt_pred
    args.model_path = 'Models/Posture/ckpts/train_syrip_withoutTrans_kpts_output_202207281246/kpts_ckpt.pth'

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Data preparation
    print('==> Preparing data..')
    testset = load_pkl_kpts_test(args.test_pred, transforms=None)  # Load dataset with only predictions
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=1)

    # Model building
    print('==> Building model..')
    net = Classifier_kpts()
    net = net.to(device)
    if device == 'mps':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Load checkpoint
    print('==> Resuming from checkpoint..')
    state_dict = torch.load(args.model_path, map_location=torch.device('mps'))['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

    # Run prediction
    pred, images, scores = test(net, device, testloader)

    return pred, images, scores
