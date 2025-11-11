import argparse
import numpy as np
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchsummary import summary

from model import *
from compute_flops import *
import math

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=400, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--refine', default='pruned.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--save', default='./finetuned', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')

parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')


def total_num_filters(mod):
    filters = 0
    for module in mod.modules():
        if isinstance(module, nn.Conv2d):
            filters = filters + module.out_channels
            #print("Number of filters inside fucntin: ", filters)
    #print("Total filters=", filters)
    return filters
device = torch.device("cuda:0" if torch.cuda.is_available() and not False else "cpu")
best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    args.distributed = args.world_size > 1

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    
    # original model imformation
    print("-----------ORIGINAL MODEL INFO-------------\n")
    model = resnet34(pretrained=False).cuda()
    summary(model, (3, 224, 224))
    flop_o_b1_false = print_model_param_flops(model=model, input_res=224, multiply_adds=False)
    print("MAC B1, False", math.ceil(flop_o_b1_false))
    flop_o_b1_true = print_model_param_flops(model=model, input_res=224, multiply_adds=True)
    print("MAC B1, True", math.ceil(flop_o_b1_true))
    filters_o=total_num_filters(model)
    print("Total number of filters", filters_o)
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters", num_parameters)
    

    print("---------\n\n ORIGINAL MODEL WITH FC=2--------------\n")
    print("\n after FC change")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    newmodel=model.cuda()

    summary(newmodel.cuda(), (3, 200, 200))

    flop_o_b1_false = print_model_param_flops(model=newmodel, input_res=200, multiply_adds=False)
    print("MAC B1, False", math.ceil(flop_o_b1_false))
    flop_o_b1_true = print_model_param_flops(model=newmodel, input_res=200, multiply_adds=True)
    print("MAC B1, True", math.ceil(flop_o_b1_true))
    filters_o=total_num_filters(newmodel)
    print("Total number of filters", filters_o)
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    print("Number of parameters", num_parameters)
    


    print("\n-------------------TRAIN ALL, FC=2, PRUNED---------------------------\n\n")
    cfg = [64, 15, 6, 128, 14, 8, 128, 256, 71, 46, 47, 51, 256, 512, 512, 512] #pr fllters left
    pmodel = resnet34(cfg=cfg).cuda()

    print("\n after FC change")
    num_ftrs = pmodel.fc.in_features
    pmodel.fc = nn.Linear(num_ftrs, 2)
    newmodel=pmodel.cuda()
    summary(pmodel.cuda(), (3, 200, 200))

    flop_p_b1_false = print_model_param_flops(model=pmodel, input_res=200, multiply_adds=False)
    print("MAC B1, False", math.ceil(flop_p_b1_false))
    flop_p_b1_true = print_model_param_flops(model=pmodel, input_res=200, multiply_adds=True)
    print("MAC B1, True", math.ceil(flop_p_b1_true))
    filters_p=total_num_filters(pmodel)
    print("Total number of filters in the pruned model", filters_p)
    num_parameters = sum([param.nelement() for param in pmodel.parameters()])
    print("Number of parameters", num_parameters)
    


if __name__ == '__main__':
    main()