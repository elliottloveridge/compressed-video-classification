from copy import deepcopy
from collections import OrderedDict
import logging
import csv
import numpy as np
from functools import partial
import os
import distiller
from distiller.scheduler import CompressionScheduler

from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
# NOTE: do I also need an import of utils?
from util import *
import test
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set

# imports
import os
import sys
import json
import numpy as np
import torch
import distiller
from datetime import datetime

# torch imports
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

# function imports
from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from util import *
from train import train_epoch
from validation import val_epoch
import test
from calculate_FLOP import model_info

msglogger = logging.getLogger()


def init_pruning(model, net_params, test_func, group):
    """perform pruning before fine-tuning on a given pre-trained model
    """

    if group not in ['element']:
        raise ValueError("group parameter contains an illegal value: {}".format(group))
    # sensitivities = OrderedDict()

    for param_name, sparsity in net_params:
        # FIXME: is this dimension check needed/correct?
        if model.state_dict()[param_name].dim() not in [2,5]:
            print('here - wrong dim')
            continue

        # model_cpy = deepcopy(model)

        if group == 'element':
            sparsity = float(sparsity)
            # element-wise sparsity pruning
            sparsity_level = {param_name: sparsity}
            pruner = distiller.pruning.SparsityLevelParameterPruner(name="sensitivity", levels=sparsity_level)
            policy = distiller.PruningPolicy(pruner, pruner_args=None)
            scheduler = CompressionScheduler(model)
            # prune only on first epoch
            scheduler.add_policy(policy, epochs=[0])

            # Compute the pruning mask per the pruner and apply the mask on the weights
            scheduler.on_epoch_begin(0)
            scheduler.mask_all_weights()

        prec1, prec5, loss = test_func(model=model)

    return prec1, prec5, loss


# get necessary args
opt = parse_opts()
if opt.root_path != '':
    opt.video_path = os.path.join(opt.root_path, opt.video_path)
    opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
    opt.result_path = os.path.join(opt.root_path, opt.result_path)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
    if opt.pretrain_path:
        opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
opt.scales = [opt.initial_scale]
for i in range(1, opt.n_scales):
    opt.scales.append(opt.scales[-1] * opt.scale_step)
opt.arch = '{}'.format(opt.model)
opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
opt.std = get_std(opt.norm_value)
torch.manual_seed(opt.manual_seed)

# load in pre-trained model
model, parameters = generate_model(opt)

criterion = nn.CrossEntropyLoss()
if not opt.no_cuda:
    criterion = criterion.cuda()

params = [('module.features.0.0.weight', 0.2),
('module.features.0.0.weight', 0.2),
('module.features.1.conv.0.weight', 0.2),
('module.features.1.conv.3.weight', 0.2),
('module.features.2.conv.0.weight', 0.2),
('module.features.2.conv.3.weight', 0.2),
('module.features.2.conv.6.weight', 0.2),
('module.features.3.conv.0.weight', 0.2),
('module.features.3.conv.3.weight', 0.2),
('module.features.3.conv.6.weight', 0.2),
('module.features.4.conv.0.weight', 0.2),
('module.features.4.conv.3.weight', 0.2),
('module.features.4.conv.6.weight', 0.2),
('module.features.5.conv.0.weight', 0.2),
('module.features.5.conv.3.weight', 0.2),
('module.features.5.conv.6.weight', 0.2),
('module.features.6.conv.0.weight', 0.2),
('module.features.6.conv.3.weight', 0.2),
('module.features.6.conv.6.weight', 0.2),
('module.features.7.conv.0.weight', 0.2),
('module.features.7.conv.3.weight', 0.2),
('module.features.7.conv.6.weight', 0.2),
('module.features.8.conv.0.weight', 0.2),
('module.features.8.conv.3.weight', 0.2),
('module.features.8.conv.6.weight', 0.2),
('module.features.9.conv.0.weight', 0.2),
('module.features.9.conv.3.weight', 0.2),
('module.features.9.conv.6.weight', 0.2),
('module.features.10.conv.0.weight', 0.2),
('module.features.10.conv.3.weight', 0.2),
('module.features.10.conv.6.weight', 0.2),
('module.features.11.conv.0.weight', 0.2),
('module.features.11.conv.3.weight', 0.2),
('module.features.11.conv.6.weight', 0.2),
('module.features.12.conv.0.weight', 0.2),
('module.features.12.conv.3.weight', 0.2),
('module.features.12.conv.6.weight', 0.2),
('module.features.13.conv.0.weight', 0.2),
('module.features.13.conv.3.weight', 0.2),
('module.features.13.conv.6.weight', 0.2),
('module.features.14.conv.0.weight', 0.2),
('module.features.14.conv.3.weight', 0.2),
('module.features.14.conv.6.weight', 0.2),
('module.features.15.conv.0.weight', 0.2),
('module.features.15.conv.3.weight', 0.2),
('module.features.15.conv.6.weight', 0.2),
('module.features.16.conv.0.weight', 0.2),
('module.features.16.conv.3.weight', 0.2),
('module.features.16.conv.6.weight', 0.2),
('module.features.17.conv.0.weight', 0.2),
('module.features.17.conv.3.weight', 0.2),
('module.features.17.conv.6.weight', 0.2),
('module.features.18.0.weight', 0.2)]

# load training and validation datasets
if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)

if opt.nesterov:
    dampening = 0
else:
    dampening = opt.dampening
optimizer = optim.SGD(
    parameters,
    lr=opt.learning_rate,
    momentum=opt.momentum,
    dampening=dampening,
    weight_decay=opt.weight_decay,
    nesterov=opt.nesterov)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=opt.lr_patience)

if not opt.no_val:
    spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    #temporal_transform = LoopPadding(opt.sample_duration)
    temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
    target_transform = ClassLabel()
    validation_data = get_validation_set(
        opt, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=16,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    val_logger = Logger(
        os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5'])

model = init_pruning(model, params, group='element')

# now test...

spatial_transform = Compose([
    Scale(int(opt.sample_size / opt.scale_in_test)),
    CornerCrop(opt.sample_size, opt.crop_position_in_test),
    ToTensor(opt.norm_value), norm_method])
# temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
# target_transform = VideoID()
target_transform = ClassLabel()

test_data = get_test_set(opt, spatial_transform, temporal_transform,
                         target_transform)

# DEBUG: the size len(test_data) might be too big
# sample a defined portion of the testing dataset
# subset_ind = np.random.randint(0, len(test_data), size=(1, 400))
# NOTE: removed .tolist() from subset_ind[0] as apparently not necessary
# test_subset = torch.utils.data.Subset(test_data, subset_ind[0])

test_loader = torch.utils.data.DataLoader(
    test_data,
    # test_data,
    batch_size=16,
    shuffle=False,
    num_workers=opt.n_threads,
    pin_memory=True)

# return the average losses, top1, top5 accuracies for subset of testing dataset
# FIXME: need to use validation.py's val_epoch instead for this
test_func = partial(test.test_eval, data_loader=test_loader, criterion=criterion, opt=opt)

prec1, prec5, loss = init_pruning(model, params, test_func, 'element')

print(prec1, prec5)
