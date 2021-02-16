from __future__ import print_function, absolute_import

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from .pose.utils.evaluation import final_preds
from .pose.utils.misc import to_numpy
from .pose.utils.osutils import isfile
from .pose.utils.transforms import fliplr, flip_back
from .pose import models as models
from .pose import datasets as datasets


# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33
# parameters
dataset = 'imgs'
njoints = 16
inp_res = 256  # input resolution
out_res = 64  # output resolution
arch = 'hg'  # model architecture
stacks = 8  # number of hourglasses to stack
blocks = 1  # number of residual modules at each location in the hourglass
resnet_layers = 50  # number of resnet layers
lr = 2.5e-4  # intial learning rate
momentum = 0.0  # momentum
weight_decay = 0.0  # weight decay (default: 0)
flip = True
resume = './Models/Hourglass/data/mpii/hg_s8_b1/model_best.pth.tar'  # load the weight from pretrained model


def hg_labeler(source):
    # create model
    print("==> creating model '{}', stacks={}, blocks={}".format(arch, stacks, blocks))
    model = models.__dict__[arch](num_stacks=stacks, num_blocks=blocks, num_classes=njoints, resnet_layers=resnet_layers)
    model = torch.nn.DataParallel(model).to(device)

    # define optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # load the weight from a specific model
    if isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        if torch.cuda.is_available():
            checkpoint = torch.load(resume)
        else:
            checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # create data loader
    imgs_dataset = datasets.__dict__[dataset](source, inp_res, out_res)
    imgs_loader = torch.utils.data.DataLoader(imgs_dataset)

    predictions = estimate(imgs_loader, model, njoints)
    preds = to_numpy(predictions)
    return preds


def estimate(imgs_loader, model, num_classes, flip=True):
    # predictions
    predictions = torch.Tensor(imgs_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, meta) in enumerate(imgs_loader):
            input = input.to(device, non_blocking=True)
            # compute output
            output = model(input)
            score_map = output[-1].cpu() if type(output) == list else output.cpu()
            if flip:
                flip_input = torch.from_numpy(fliplr(input.cpu().clone().numpy())).float().to(device)
                flip_output = model(flip_input)
                flip_output = flip_output[-1].cpu() if type(flip_output) == list else flip_output.cpu()
                flip_output = flip_back(flip_output)
                score_map += flip_output

            # generate predictions
            preds = final_preds(score_map, meta['center'], meta['scale'], [out_res, out_res])
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

    return predictions

