from copy import deepcopy
from collections import OrderedDict
import logging
import csv
import numpy as np
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

torch.manual_seed(opt.manual_seed)

model, parameters = generate_model(opt)
# print(model)

criterion = nn.CrossEntropyLoss()
if not opt.no_cuda:
    criterion = criterion.cuda()

if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)

# FIXME: this shouldn't be hardcoded! - perform a model summary here
# FIXME: want to do it without adding 'module.module.'
params = ['module.module.features.0.0.weight',
'module.module.features.0.1.weight',
'module.module.features.1.conv.0.weight',
'module.module.features.1.conv.1.weight',
'module.module.features.1.conv.3.weight',
'module.module.features.1.conv.4.weight',
'module.module.features.2.conv.0.weight',
'module.module.features.2.conv.1.weight',
'module.module.features.2.conv.3.weight',
'module.module.features.2.conv.4.weight',
'module.module.features.2.conv.6.weight',
'module.module.features.2.conv.7.weight',
'module.module.features.3.conv.0.weight',
'module.module.features.3.conv.1.weight',
'module.module.features.3.conv.3.weight',
'module.module.features.3.conv.4.weight',
'module.module.features.3.conv.6.weight',
'module.module.features.3.conv.7.weight',
'module.module.features.4.conv.0.weight',
'module.module.features.4.conv.1.weight',
'module.module.features.4.conv.3.weight',
'module.module.features.4.conv.4.weight',
'module.module.features.4.conv.6.weight',
'module.module.features.4.conv.7.weight',
'module.module.features.5.conv.0.weight',
'module.module.features.5.conv.1.weight',
'module.module.features.5.conv.3.weight',
'module.module.features.5.conv.4.weight',
'module.module.features.5.conv.6.weight',
'module.module.features.5.conv.7.weight',
'module.module.features.6.conv.0.weight',
'module.module.features.6.conv.1.weight',
'module.module.features.6.conv.3.weight',
'module.module.features.6.conv.4.weight',
'module.module.features.6.conv.6.weight',
'module.module.features.6.conv.7.weight',
'module.module.features.7.conv.0.weight',
'module.module.features.7.conv.1.weight',
'module.module.features.7.conv.3.weight',
'module.module.features.7.conv.4.weight',
'module.module.features.7.conv.6.weight',
'module.module.features.7.conv.7.weight',
'module.module.features.8.conv.0.weight',
'module.module.features.8.conv.1.weight',
'module.module.features.8.conv.3.weight',
'module.module.features.8.conv.4.weight',
'module.module.features.8.conv.6.weight',
'module.module.features.8.conv.7.weight',
'module.module.features.9.conv.0.weight',
'module.module.features.9.conv.1.weight',
'module.module.features.9.conv.3.weight',
'module.module.features.9.conv.4.weight',
'module.module.features.9.conv.6.weight',
'module.module.features.9.conv.7.weight',
'module.module.features.10.conv.0.weight',
'module.module.features.10.conv.1.weight',
'module.module.features.10.conv.3.weight',
'module.module.features.10.conv.4.weight',
'module.module.features.10.conv.6.weight',
'module.module.features.10.conv.7.weight',
'module.module.features.11.conv.0.weight',
'module.module.features.11.conv.1.weight',
'module.module.features.11.conv.3.weight',
'module.module.features.11.conv.4.weight',
'module.module.features.11.conv.6.weight',
'module.module.features.11.conv.7.weight',
'module.module.features.12.conv.0.weight',
'module.module.features.12.conv.1.weight',
'module.module.features.12.conv.3.weight',
'module.module.features.12.conv.4.weight',
'module.module.features.12.conv.6.weight',
'module.module.features.12.conv.7.weight',
'module.module.features.13.conv.0.weight',
'module.module.features.13.conv.1.weight',
'module.module.features.13.conv.3.weight',
'module.module.features.13.conv.4.weight',
'module.module.features.13.conv.6.weight',
'module.module.features.13.conv.7.weight',
'module.module.features.14.conv.0.weight',
'module.module.features.14.conv.1.weight',
'module.module.features.14.conv.3.weight',
'module.module.features.14.conv.4.weight',
'module.module.features.14.conv.6.weight',
'module.module.features.14.conv.7.weight',
'module.module.features.15.conv.0.weight',
'module.module.features.15.conv.1.weight',
'module.module.features.15.conv.3.weight',
'module.module.features.15.conv.4.weight',
'module.module.features.15.conv.6.weight',
'module.module.features.15.conv.7.weight',
'module.module.features.16.conv.0.weight',
'module.module.features.16.conv.1.weight',
'module.module.features.16.conv.3.weight',
'module.module.features.16.conv.4.weight',
'module.module.features.16.conv.6.weight',
'module.module.features.16.conv.7.weight',
'module.module.features.17.conv.0.weight',
'module.module.features.17.conv.1.weight',
'module.module.features.17.conv.3.weight',
'module.module.features.17.conv.4.weight',
'module.module.features.17.conv.6.weight',
'module.module.features.17.conv.7.weight',
'module.module.features.18.0.weight',
'module.module.features.18.1.weight',
'module.module.classifier.1.weight']

best_prec1 = 0
if opt.resume_path:
    # NOTE: is dataparallel needed?
    model = nn.DataParallel(model)
    # # NOTE: is this needed? - less likely than dataparallel
    # model = model.cuda()
    print('loading checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    # NOTE: create new OrderedDict with additional `module.`
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        # NOTE: this is hacky, remove it and get working without
        name = 'module.' + k
        new_state_dict[name] = v
    assert opt.arch == checkpoint['arch']
    best_prec1 = checkpoint['best_prec1']
    opt.begin_epoch = checkpoint['epoch']

    model.load_state_dict(new_state_dict)

# introduce a range of sparsity values
# sensitivities = np.arange(*args.sensitivity_range)
sparse_rng = range(0, 1, 10)

# same as test call in main.py but includes sampling
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
subset_ind = np.random.randint(0, len(test_data), size=(1, 2))
# NOTE: removed .tolist() from subset_ind[0] as apparently not necessary
test_subset = torch.utils.data.Subset(test_data, subset_ind[0])

test_loader = torch.utils.data.DataLoader(
    test_subset,
    # test_data,
    batch_size=2,
    shuffle=False,
    num_workers=opt.n_threads,
    pin_memory=True)

# NOTE: is this needed?
criterion = nn.CrossEntropyLoss()
if not opt.no_cuda:
    criterion = criterion.cuda()

# return the average losses, top1, top5 accuracies for subset of testing dataset
# FIXME: need to use validation.py's val_epoch instead for this
# test_func = test.test_eval(test_loader, model, opt, test_data.class_names, criterion)

# test_logger = Logger(
#     os.path.join(opt.result_path, 'test.log'),
#     ['loss', 'prec1', 'prec5'])

test_func = test.test_eval(test_loader, model, criterion, opt)

# group='filter' used to define filter pruning
# FIXME: the 'filter' command should be editable

sense = perform_sensitivity_analysis(model, params, sparsities=sparse_rng,
test_func=test_func, group='filter')
sensitivities_to_png(sense, os.path.join(opt.result_path,'sensitivity.png'))
sensitivities_to_csv(sense, os.path.join(opt.result_path, 'sensitivity.csv'))
