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
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
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
            #RandomRotate(),
            #RandomResize(),
            crop_method,
            #MultiplyValues(),
            #Dropout(),
            #SaltImage(),
            #Gaussian_blur(),
            #SpatialElasticDisplacement(),
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

    best_prec1 = 0
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    # set compression dictionary to validate compression_type input
    # FIXME: move this to somewhere else, not a great implementation
    comp = dict()
    # active compression = element-wise pruning and quantisation aware training
    comp['active'] = ['qat, ep']
    comp['passive'] = ['ptq']

    if opt.compression_type == 'ptq':
        # classifier.acts_quant_stats_collection(model, criterion, pylogger, args, save_to_file=True)
        print('here')

    opt.kd_policy = None

    if opt.compress:
        compression_scheduler = distiller.CompressionScheduler(model)
        compression_scheduler = distiller.file_config(model, optimizer, opt.compression_file, compression_scheduler)

        spar = sum(distiller.utils.sparsity(p.data) for p in model.parameters() if p.requires_grad)

        print('initial sum of weight sparsity:', spar)

    else:
        compression_scheduler = None

    print('run')

    for i in range(opt.begin_epoch, opt.begin_epoch + opt.n_epochs):

        if opt.compression_type in comp['active'] and opt.compress:
            compression_scheduler.on_epoch_begin(i)

        if not opt.no_train:

            adjust_learning_rate(optimizer, i, opt)
            # train_epoch(i, train_loader, model, criterion, optimizer, opt,
            #             train_logger, train_batch_logger, compression_scheduler)

            # def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
            #                 epoch_logger, batch_logger, compression_scheduler):

            print('train at epoch {}'.format(i))

            model.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            end_time = time.time()

            for j, (inputs, targets) in enumerate(train_loader):
                data_time.update(time.time() - end_time)

                if compression_scheduler is not None:
                    compression_scheduler.on_minibatch_begin(i, minibatch_id=j,
                    minibatches_per_epoch=len(train_loader))

                if not opt.no_cuda:
                    targets = targets.cuda()
                inputs = Variable(inputs)
                targets = Variable(targets)
                if opt.kd_policy is None:
                    # Revert to a "normal" forward-prop call if no knowledge distillation policy is present
                    outputs = model(inputs)
                else:
                    # kd forward pass through
                    outputs = opt.kd_policy.forward(inputs)

                loss = criterion(outputs, targets)

                # before backwards pass - update loss to include regularization
                d_loss = compression_scheduler.before_backward_pass(i, minibatch_id=j,
                minibatches_per_epoch=len(train_loader), loss=loss)

                losses.update(d_loss.data, inputs.size(0))
                prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
                top1.update(prec1, inputs.size(0))
                top5.update(prec5, inputs.size(0))

                optimizer.zero_grad()
                d_loss.backward()

                compression_scheduler.before_parameter_optimization(i, minibatch_id=j,
                minibatches_per_epoch=len(train_loader), optimizer=optimizer)

                optimizer.step()

                compression_scheduler.on_minibatch_end(i, minibatch_id=j, minibatches_per_epoch=len(train_loader))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                train_batch_logger.log({
                    'epoch': i,
                    'batch': j + 1,
                    'iter': (i - 1) * len(train_loader) + (j + 1),
                    'loss': losses.val.item(),
                    'prec1': top1.val.item(),
                    'prec5': top5.val.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })
                if i % 10 ==0:
                    print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                          'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                              epoch,
                              i,
                              len(train_loader),
                              batch_time=batch_time,
                              data_time=data_time,
                              loss=losses,
                              top1=top1,
                              top5=top5,
                              lr=optimizer.param_groups[0]['lr']))

            train_logger.log({
                'epoch': i,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item(),
                'lr': optimizer.param_groups[0]['lr']
            })

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

        if opt.compression_type in comp['active'] and opt.compress:
            compression_scheduler.on_epoch_end(i)

    if opt.compression_type == 'ptq':
        quantizer = distiller.quantization.PostTrainLinearQuantizer(model, bits_activations=None, bits_weights=8)
        # NOTE: need to add the input shape!
        quantizer.prepare_model(torch.rand(1, 3, 16, 112, 112))
        # NOTE: should the model be saved here?

    # print flops and params to check for reduction
    if opt.compress:
        # par = sum(p.numel() - p.nonzero().size(0) for p in model.parameters() if p.requires_grad)
        # print("post-compression zero parameter count: ", par)

        spar = sum(distiller.utils.sparsity(p.data) for p in model.parameters() if p.requires_grad)
        print('sum of weight sparsity:', spar)

        for p in model.named_parameters():
            print(p)

        # NOTE: FLOPs not working
        # par, flo = model_info(model, opt)
        # print('Post Compression:')
        # print('Trainiable Parameters:', par)
        # print('FLOPs:', flo)

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method])
        # temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
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