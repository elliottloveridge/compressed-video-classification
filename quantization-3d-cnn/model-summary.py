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
from utils import *

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
    if opt.pretrain_path:
        opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
opt.scales = [opt.initial_scale]
for i in range(1, opt.n_scales):
    opt.scales.append(opt.scales[-1] * opt.scale_step)
opt.arch = '{}'.format(opt.model)
# opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
# opt.std = get_std(opt.norm_value)
opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.width_mult) + 'x',
                           opt.modality, str(opt.sample_duration)])

# print(opt)
with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
    json.dump(vars(opt), opt_file)

torch.manual_seed(opt.manual_seed)

model, parameters = generate_model(opt)
print(model)

df = distiller.model_summary(model, None, dataset='ucf101')

# from distiller example jupyter notebooks...
dummy_input = torch.randn(1, 3, 224, 224)
ms = distiller.model_performance_summary(model, dummy_input, 1)

df.to_csv(opt.result_path)

ms.to_csv(opt.result_path)
