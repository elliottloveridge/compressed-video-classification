import os
import sys
import json
import numpy as np
import torch
import pandas as pd

import distiller

from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from util import *
import test
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set

from calculate_FLOP import model_info

# opts kept the same even if not needed
opt = parse_opts()
if opt.root_path != '':
    opt.video_path = os.path.join(opt.root_path, opt.video_path)
    opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
    opt.result_path = os.path.join(opt.root_path, opt.result_path)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
opt.arch = '{}'.format(opt.model)
opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
opt.std = get_std(opt.norm_value)
# NOTE: removed opt.store_name arg from here

# NOTE: added for norm_method used in test - need to check what it does
if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)

torch.manual_seed(opt.manual_seed)

model, parameters = generate_model(opt)

best_prec1 = 0
if opt.resume_path:
    print('loading checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    assert opt.arch == checkpoint['arch']
    best_prec1 = checkpoint['best_prec1']
    opt.begin_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

name_ = []
module_ = []

for name, module in model.named_modules():
    if len(module._modules) == 0:
        # FIXME: don't want this hardcoded
        if module.__class__.__name__ not in ['ReLU', 'ReLU6', 'MaxPool3d', 'AvgPool3d', 'Dropout', 'Sequential', 'AdaptiveAvgPool3d']:
            module_.append(module.__class__.__name__)

for name, state in model.named_parameters():
    if name[-6:] == 'weight':
        name_.append(name)

df = pd.DataFrame({'Name': name_, 'Type': module_})

print(df)

f = opt.arch
if opt.arch in ['resnet', 'csn']:
    f += str(opt.model_depth)
f += '.csv'

f = os.path.join('/app/compressed-3d-cnn/model_summary/', f)

print('saved to file:', f)

df.to_csv(f, index=False)
