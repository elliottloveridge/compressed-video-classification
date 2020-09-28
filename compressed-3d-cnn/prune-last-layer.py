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

# FIXME: new import
# from new-prune import init_pruning


if __name__ == '__main__':

    date_str = datetime.today().strftime('%d%m')

    # define input args
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
    opt.t_arch = '{}'.format(opt.t_model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    if opt.compression_type != '':
        # FIXME: added test ep comperssion type
        assert opt.compression_type in ['ep', 'ptq', 'qat', 'kd', 'ep-test']
    if not opt.compress:
        opt.compression_type = 'benchmark'
    opt.store_name = '_'.join([opt.dataset, opt.model, opt.compression_type,
    str(opt.n_epochs) + 'epochs', date_str])

    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)

    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            RandomHorizontalFlip(),
            crop_method,
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

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

    best_prec1 = 0

    # don't currently allow for resumed training whilst using kd
    if opt.resume_path and opt.compression_type != 'kd':
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    # %% start of pruning test

    def init_pruning(model, net_params, group):
        """perform pruning before fine-tuning on a given pre-trained model
        """

        if group not in ['element']:
            raise ValueError("group parameter contains an illegal value: {}".format(group))

        for param_name, sparsity in net_params:
            if model.state_dict()[param_name].dim() not in [2,5]:
                continue

            if group == 'element':
                sparsity = float(sparsity)
                # element-wise sparsity pruning
                sparsity_level = {param_name: sparsity}
                pruner = distiller.pruning.SparsityLevelParameterPruner(name="sensitivity", levels=sparsity_level)
                policy = distiller.PruningPolicy(pruner, pruner_args=None)
                scheduler = CompressionScheduler(model)
                # FIXME: this may not prune properly
                print(opt.begin_epoch)
                scheduler.add_policy(policy, epochs=[opt.begin_epoch])

                # Compute the pruning mask per the pruner and apply the mask on the weights
                scheduler.on_epoch_begin(0)
                scheduler.mask_all_weights()

        return model

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

    ('pruning model')
    model = init_pruning(model, params, group='element')
    par = sum(p.numel() - p.nonzero().size(0) for p in model.parameters() if p.requires_grad)
    print("post-compression zero parameter count:", par)

    # %% end of pruning test

    if not opt.no_train:

        if opt.compression_type == 'ptq':
            # FIXME: add stats collection here
            # classifier.acts_quant_stats_collection(model, criterion, pylogger, args, save_to_file=True)
            print('add stats file here')

        if opt.compress and opt.compression_type in ['qat', 'ep', 'kd']:
            compression_scheduler = distiller.CompressionScheduler(model)
            if opt.compression_type in ['qat', 'ep']:
                compression_scheduler = distiller.file_config(model, optimizer, opt.compression_file, compression_scheduler)

            # get initial sparsity sum as a test for pruning
            if opt.compression_type == 'ep':
                par = sum(p.numel() - p.nonzero().size(0) for p in model.parameters() if p.requires_grad)
                print("pre-compression zero parameter count:", par)

        else:
            compression_scheduler = None

    # set distillation policy as None
    opt.kd_policy = None

    if opt.compress and opt.compression_type == 'kd':

        # generate teacher model and load state_dict
        teacher, parameters = generate_model(opt, teacher=True)
        print('loading checkpoint {}'.format(opt.t_path))
        checkpoint = torch.load(opt.t_path)
        assert opt.t_arch == checkpoint['arch']
        teacher.load_state_dict(checkpoint['state_dict'])

        # create a distillation policy and add to compression_scheduler
        dlw = distiller.DistillationLossWeights(opt.kd_distill_wt,
              opt.kd_student_wt, opt.kd_teacher_wt)
        opt.kd_policy = distiller.KnowledgeDistillationPolicy(model,
                        teacher, opt.kd_temp, dlw)
        # FIXME: this may not be correct
        end_epoch = opt.begin_epoch + opt.n_epochs
        compression_scheduler.add_policy(opt.kd_policy,
                                        starting_epoch=opt.begin_epoch,
                                        ending_epoch=end_epoch, frequency=1)


    print('run')

    for i in range(opt.begin_epoch, opt.begin_epoch + opt.n_epochs):

        if opt.compress and opt.compression_type in ['qat', 'ep']:
            compression_scheduler.on_epoch_begin(i)

        if not opt.no_train:

            adjust_learning_rate(optimizer, i, opt)
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger, compression_scheduler)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, False, opt)

        if not opt.no_val:

            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, is_best, opt)

        if opt.compress and opt.compression_type in ['qat', 'ep']:
            compression_scheduler.on_epoch_end(i)

    if opt.compression_type == 'ptq':
        quantizer = distiller.quantization.PostTrainLinearQuantizer(model, bits_activations=1, bits_parameters=1)
        quantizer.prepare_model(torch.rand(1, 3, 16, 112, 112))
        # NOTE: should the model be saved here?

    # test for parameter reduction
    if opt.compress:

        # # FIXME: this may not be working
        # prms, flps = model_info(model, opt)
        # print("number of flops:", flps)

        par = sum(p.numel() - p.nonzero().size(0) for p in model.parameters() if p.requires_grad)
        print("post-finetune zero parameter count:", par)

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method])
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=16,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
