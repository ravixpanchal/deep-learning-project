from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import math

import argparse
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
from torch.autograd import Variable
#from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from compute_flops import *
from model import *
#from utils import * #getloader, accuracy, save_checkpoint, save_checkpoint_pruned

# Prune settings
parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')
parser.add_argument('--data', type=str, default='/scratch/zhuangl/datasets/imagenet',
                    help='Path to imagenet validation data')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')
parser.add_argument('-v', default='A', type=str, 
                    help='version of the pruned model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def imagenetdataset():
	# Data loading code
    traindir = os.path.join('', '/storage/research/data/Tejalal/ImageNetDataset/train')
    valdir = os.path.join('', '/storage/research/data/Tejalal/ImageNetDataset/val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, 
        	transforms.Compose([
	            transforms.Resize(256),
	            transforms.CenterCrop(224),
	            transforms.ToTensor(),
	            normalize,
        ])),
        batch_size=8, shuffle=False,
        num_workers=0, pin_memory=True)
    return train_loader, test_loader

def prune_model(model, f1, f2, f3):
  skip = {
    'A': [3, 12, 24, 39, 42, 45, 48], # [2, 8, 14, 16, 26, 28, 30, 32],
    'B': [2, 8, 14, 16, 26, 28, 30, 32],
  }
  f1=round(f1, 2)
  f2=round(f2, 2)
  f3=round(f3, 2)
  prune_prob = {
      'A': [f1, f2, f3, 0.0],
      'B': [0.5, 0.6, 0.4, 0.0],
  }

  layer_id = 1
  cfg = []
  cfg_mask = []
  for m in model.modules():
      if isinstance(m, nn.Conv2d):
          if m.kernel_size == (1,1):
              continue
          out_channels = m.weight.data.shape[0]
          if layer_id in skip[args.v]:
              cfg_mask.append(torch.ones(out_channels))
              cfg.append(out_channels)
              layer_id += 1
              continue
          if layer_id % 2 == 0:
              if layer_id <= 9: #6
                  stage = 0
              elif layer_id <= 21: #14
                  stage = 1
              elif layer_id <= 26: #39
                  stage = 2
              else:
                  stage = 3
              prune_prob_stage = prune_prob[args.v][stage]
              print("PRUNE PROB STAGE", prune_prob_stage)
              weight_copy = m.weight.data.abs().clone().cpu().numpy()
              L1_norm = np.sum(weight_copy, axis=(1,2,3))
              num_keep = int(out_channels * (1 - prune_prob_stage))
              arg_max = np.argsort(L1_norm)
              arg_max_rev = arg_max[::-1][:num_keep]
              mask = torch.zeros(out_channels)
              mask[arg_max_rev.tolist()] = 1
              cfg_mask.append(mask)
              print("layer mask", len(cfg_mask))
              cfg.append(num_keep)
              layer_id += 1
              continue
          layer_id += 1

  print("PR_CFG", cfg)
  assert len(cfg) == 16, "Length of cfg variable is not correct."
  
  newmodel = resnet34(cfg=cfg)
  newmodel = torch.nn.DataParallel(newmodel).cuda()

  start_mask = torch.ones(3)
  layer_id_in_cfg = 0
  conv_count = 1
  for [m0, m1] in zip(model.modules(), newmodel.modules()):
      if isinstance(m0, nn.Conv2d):
          if m0.kernel_size == (1,1):
              # Cases for down-sampling convolution.
              m1.weight.data = m0.weight.data.clone()
              continue
          if conv_count == 1:
              m1.weight.data = m0.weight.data.clone()
              conv_count += 1
              continue
          if conv_count % 2 == 0:
              mask = cfg_mask[layer_id_in_cfg]
              idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
              if idx.size == 1:
                  idx = np.resize(idx, (1,))
              w = m0.weight.data[idx.tolist(), :, :, :].clone()
              m1.weight.data = w.clone()
              layer_id_in_cfg += 1
              conv_count += 1
              continue
          if conv_count % 2 == 1:
              mask = cfg_mask[layer_id_in_cfg-1]
              idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
              if idx.size == 1:
                  idx = np.resize(idx, (1,))
              w = m0.weight.data[:, idx.tolist(), :, :].clone()
              m1.weight.data = w.clone()
              conv_count += 1
              continue
      elif isinstance(m0, nn.BatchNorm2d):
          assert isinstance(m1, nn.BatchNorm2d), "There should not be bn layer here."
          if conv_count % 2 == 1:
              mask = cfg_mask[layer_id_in_cfg-1]
              idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
              if idx.size == 1:
                  idx = np.resize(idx, (1,))
              m1.weight.data = m0.weight.data[idx.tolist()].clone()
              m1.bias.data = m0.bias.data[idx.tolist()].clone()
              m1.running_mean = m0.running_mean[idx.tolist()].clone()
              m1.running_var = m0.running_var[idx.tolist()].clone()
              continue
          m1.weight.data = m0.weight.data.clone()
          m1.bias.data = m0.bias.data.clone()
          m1.running_mean = m0.running_mean.clone()
          m1.running_var = m0.running_var.clone()
      elif isinstance(m0, nn.Linear):
          m1.weight.data = m0.weight.data.clone()
          m1.bias.data = m0.bias.data.clone()

  #torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
  return newmodel, cfg