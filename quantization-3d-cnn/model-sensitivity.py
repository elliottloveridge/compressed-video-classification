#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Perform sensitivity tests on layers and whole networks.

Construct a schedule for experimenting with network and layer sensitivity
to pruning.

The idea is to set the pruning level (percentage) of specific layers (or the
entire network), and then to prune once, run an evaluation on the test dataset,
and exit.  This should teach us about the "sensitivity" of the network/layers
to pruning.

This concept is discussed in "Learning both Weights and Connections for
Efficient Neural Networks" - https://arxiv.org/pdf/1506.02626v3.pdf
"""

from copy import deepcopy
from collections import OrderedDict
import logging
import csv
import numpy as np
import os
import distiller
# NOTE: edited, check it is working correctly
from distiller.scheduler import CompressionScheduler

from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from utils import *
import test
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set

msglogger = logging.getLogger()


def perform_sensitivity_analysis(model, net_params, sparsities, test_func, group):
    """Perform a sensitivity test for a model's weights parameters.

    The model should be trained to maximum accuracy, because we aim to understand
    the behavior of the model's performance in relation to pruning of a specific
    weights tensor.

    By default this function will test all of the model's parameters.

    The return value is a complex sensitivities dictionary: the dictionary's
    key is the name (string) of the weights tensor.  The value is another dictionary,
    where the tested sparsity-level is the key, and a (top1, top5, loss) tuple
    is the value.
    Below is an example of such a dictionary:

    .. code-block:: python
    {'features.module.6.weight':    {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.492, 79.1,   1.9161),
                                     0.10: (56.212, 78.854, 1.9315),
                                     0.15: (35.424, 60.3,   3.0866)},
     'classifier.module.1.weight':  {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.514, 79.07,  1.9159),
                                     0.10: (56.434, 79.074, 1.9138),
                                     0.15: (54.454, 77.854, 2.3127)} }

    The test_func is expected to execute the model on a test/validation dataset,
    and return the results for top1 and top5 accuracies, and the loss value.
    """
    if group not in ['element', 'filter', 'channel']:
        raise ValueError("group parameter contains an illegal value: {}".format(group))
    sensitivities = OrderedDict()

    for param_name in net_params:
        if model.state_dict()[param_name].dim() not in [2,4]:
            continue

        # Make a copy of the model, because when we apply the zeros mask (i.e.
        # perform pruning), the model's weights are altered
        model_cpy = deepcopy(model)

        sensitivity = OrderedDict()
        for sparsity_level in sparsities:
            sparsity_level = float(sparsity_level)
            msglogger.info("Testing sensitivity of %s [%0.1f%% sparsity]" % (param_name, sparsity_level*100))
            # Create the pruner (a level pruner), the pruning policy and the
            # pruning schedule.
            if group == 'element':
                # Element-wise sparasity
                sparsity_levels = {param_name: sparsity_level}
                pruner = distiller.pruning.SparsityLevelParameterPruner(name="sensitivity", levels=sparsity_levels)
            elif group == 'filter':
                # Filter ranking
                if model.state_dict()[param_name].dim() != 4:
                    continue
                pruner = distiller.pruning.L1RankedStructureParameterPruner("sensitivity",
                                                                            group_type="Filters",
                                                                            desired_sparsity=sparsity_level,
                                                                            weights=param_name)
            elif group == 'channel':
                # Filter ranking
                if model.state_dict()[param_name].dim() != 4:
                    continue
                pruner = distiller.pruning.L1RankedStructureParameterPruner("sensitivity",
                                                                            group_type="Channels",
                                                                            desired_sparsity=sparsity_level,
                                                                            weights=param_name)

            policy = distiller.PruningPolicy(pruner, pruner_args=None)
            scheduler = CompressionScheduler(model_cpy)
            scheduler.add_policy(policy, epochs=[0])

            # Compute the pruning mask per the pruner and apply the mask on the weights
            scheduler.on_epoch_begin(0)
            scheduler.mask_all_weights()

            # Test and record the performance of the pruned model
            prec1, prec5, loss = test_func(model=model_cpy)
            sensitivity[sparsity_level] = (prec1, prec5, loss)
            sensitivities[param_name] = sensitivity
    return sensitivities


def sensitivities_to_png(sensitivities, fname):
    """Create a mulitplot of the sensitivities.

    The 'sensitivities' argument is expected to have the dict-of-dict structure
    described in the documentation of perform_sensitivity_test.
    """
    try:
        # sudo apt-get install python3-tk
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: Function plot_sensitivity requires package matplotlib which"
              "is not installed in your execution environment.\n"
              "Skipping the PNG file generation")
        return

    msglogger.info("Generating sensitivity graph")

    for param_name, sensitivity in sensitivities.items():
        sense = [values[1] for sparsity, values in sensitivity.items()]
        sparsities = [sparsity for sparsity, values in sensitivity.items()]
        plt.plot(sparsities, sense, label=param_name)

    plt.ylabel('top5')
    plt.xlabel('sparsity')
    plt.title('Pruning Sensitivity')
    plt.legend(loc='lower center',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(fname, format='png')


def sensitivities_to_csv(sensitivities, fname):
    """Create a CSV file listing from the sensitivities dictionary.

    The 'sensitivities' argument is expected to have the dict-of-dict structure
    described in the documentation of perform_sensitivity_test.
    """
    with open(fname, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # write the header
        writer.writerow(['parameter', 'sparsity', 'top1', 'top5', 'loss'])
        for param_name, sensitivity in sensitivities.items():
            for sparsity, values in sensitivity.items():
                writer.writerow([param_name] + [sparsity] + list(values))


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

# NOTE: you can add the classifiers to pruning params, these are fully connected layers?
params = ['module.features.0.0',
'module.features.0.1',
'module.features.0.2',
'module.features.1.conv.0',
'module.features.1.conv.1',
'module.features.1.conv.2',
'module.features.1.conv.3',
'module.features.1.conv.4',
'module.features.2.conv.0',
'module.features.2.conv.1',
'module.features.2.conv.2',
'module.features.2.conv.3',
'module.features.2.conv.4',
'module.features.2.conv.5',
'module.features.2.conv.6',
'module.features.2.conv.7',
'module.features.3.conv.0',
'module.features.3.conv.1',
'module.features.3.conv.2',
'module.features.3.conv.3',
'module.features.3.conv.4',
'module.features.3.conv.5',
'module.features.3.conv.6',
'module.features.3.conv.7',
'module.features.4.conv.0',
'module.features.4.conv.1',
'module.features.4.conv.2',
'module.features.4.conv.3',
'module.features.4.conv.4',
'module.features.4.conv.5',
'module.features.4.conv.6',
'module.features.4.conv.7',
'module.features.5.conv.0',
'module.features.5.conv.1',
'module.features.5.conv.2',
'module.features.5.conv.3',
'module.features.5.conv.4',
'module.features.5.conv.5',
'module.features.5.conv.6',
'module.features.5.conv.7',
'module.features.6.conv.0',
'module.features.6.conv.1',
'module.features.6.conv.2',
'module.features.6.conv.3',
'module.features.6.conv.4',
'module.features.6.conv.5',
'module.features.6.conv.6',
'module.features.6.conv.7',
'module.features.7.conv.0',
'module.features.7.conv.1',
'module.features.7.conv.2',
'module.features.7.conv.3',
'module.features.7.conv.4',
'module.features.7.conv.5',
'module.features.7.conv.6',
'module.features.7.conv.7',
'module.features.8.conv.0',
'module.features.8.conv.1',
'module.features.8.conv.2',
'module.features.8.conv.3',
'module.features.8.conv.4',
'module.features.8.conv.5',
'module.features.8.conv.6',
'module.features.8.conv.7',
'module.features.9.conv.0',
'module.features.9.conv.1',
'module.features.9.conv.2',
'module.features.9.conv.3',
'module.features.9.conv.4',
'module.features.9.conv.5',
'module.features.9.conv.6',
'module.features.9.conv.7',
'module.features.10.conv.0',
'module.features.10.conv.1',
'module.features.10.conv.2',
'module.features.10.conv.3',
'module.features.10.conv.4',
'module.features.10.conv.5',
'module.features.10.conv.6',
'module.features.10.conv.7',
'module.features.11.conv.0',
'module.features.11.conv.1',
'module.features.11.conv.2',
'module.features.11.conv.3',
'module.features.11.conv.4',
'module.features.11.conv.5',
'module.features.11.conv.6',
'module.features.11.conv.7',
'module.features.12.conv.0',
'module.features.12.conv.1',
'module.features.12.conv.2',
'module.features.12.conv.3',
'module.features.12.conv.4',
'module.features.12.conv.5',
'module.features.12.conv.6',
'module.features.12.conv.7',
'module.features.13.conv.0',
'module.features.13.conv.1',
'module.features.13.conv.2',
'module.features.13.conv.3',
'module.features.13.conv.4',
'module.features.13.conv.5',
'module.features.13.conv.6',
'module.features.13.conv.7',
'module.features.14.conv.0',
'module.features.14.conv.1',
'module.features.14.conv.2',
'module.features.14.conv.3',
'module.features.14.conv.4',
'module.features.14.conv.5',
'module.features.14.conv.6',
'module.features.14.conv.7',
'module.features.15.conv.0',
'module.features.15.conv.1',
'module.features.15.conv.2',
'module.features.15.conv.3',
'module.features.15.conv.4',
'module.features.15.conv.5',
'module.features.15.conv.6',
'module.features.15.conv.7',
'module.features.16.conv.0',
'module.features.16.conv.1',
'module.features.16.conv.2',
'module.features.16.conv.3',
'module.features.16.conv.4',
'module.features.16.conv.5',
'module.features.16.conv.6',
'module.features.16.conv.7',
'module.features.17.conv.0',
'module.features.17.conv.1',
'module.features.17.conv.2',
'module.features.17.conv.3',
'module.features.17.conv.4',
'module.features.17.conv.5',
'module.features.17.conv.6',
'module.features.17.conv.7',
'module.features.18.0',
'module.features.18.1',
'module.features.18.2']

best_prec1 = 0
if opt.resume_path:
    print('loading checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    assert opt.arch == checkpoint['arch']
    best_prec1 = checkpoint['best_prec1']
    opt.begin_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

# sparsities = should be a range of values to perform sparsity calculations on
# test_func = should be a function that returns the average loss + evaluation metric (top1/5 accuracy) from a portion of your test dataset (rand?)
# group = 'filter' - if using filter wise pruning etc

sparse_rng = range(0, 1, 5)

# #------------------- data sampler start ------------------#
# # FIXME: need to move this class to the top of the page...
#
# class DataSampler():
#     def __init__(self, mask):
#         self.mask = mask
#
#     def __iter__(self):
#         return (self.indices[i] for i in torch.nonzero(self.mask))
#
#     def __len__(self):
#         return len(self.mask)
#
# #------------------- data sampler end ------------------#


# # distiller code example - do I need loggers etc?
# test_fnc = partial(classifier.test, test_loader=data_loader, criterion=criterion,
#                        loggers=loggers, args=args,
#                        activations_collectors=classifier.create_activation_stats_collectors(model))


# # sample only a select number of values from test dataset
# sampler = DataSampler(100)

# same as test call in main.py but includes sampling
spatial_transform = Compose([
    Scale(int(opt.sample_size / opt.scale_in_test)),
    CornerCrop(opt.sample_size, opt.crop_position_in_test),
    ToTensor(opt.norm_value), norm_method])
# temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
target_transform = VideoID()

test_data = get_test_set(opt, spatial_transform, temporal_transform,
                         target_transform)

# NOTE: using test_subset will negate the need for sampler in DataLoader

# NOTE: the size len(test_data) might be too big
subset_ind = np.random.randint(0, len(test_data), size=(1, 100))

print(subset_ind[0].tolist())
print()
print(type(subset_ind))

test_subset = torch.utils.data.Subset(test_data, subset_ind[0].tolist())

test_loader = torch.utils.data.DataLoader(
    # test_data,
    test_subset,
    batch_size=16,
    shuffle=False,
    # sampler=sampler,
    num_workers=opt.n_threads,
    pin_memory=True)

test_func = test.test_eval(test_loader, model, opt, test_data.class_names)

sense = perform_sensitivity_analysis(model, params, sparsities=sparse_rng, test_func=test_func, group='filter')
sensitivities_to_png(sense, opt.result_path + 'sensitivity.png')
sensitivities_to_csv(sense, opt.result_path + 'sensitivity.csv')
