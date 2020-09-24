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


# NOTE: this is a test file for knowledge distillation, will eventually be merged into main
# after it is confirmed that kd will work from the distiller package.


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
    # NOTE: added this for distillation
    opt.t_arch = '{}'.format(opt.t_model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    if not opt.compress:
        opt.compression_type = 'benchmark'
    opt.store_name = '_'.join([opt.dataset, opt.model, opt.compression_type,
    str(opt.n_epochs) + 'epochs', date_str])

    # save results as a text file
    path = os.path.join(opt.result_path, 'accuracy.txt')

    for opt.kd_temp in [5]:
        for opt.kd_distill_wt in [0.8]:
            for opt.kd_student_wt in [0.3, 0.4]:

                print("testing for temp={}, distill_wt={}, student_wt={}".format(opt.kd_temp, opt.kd_distill_wt, opt.kd_student_wt))

                with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
                    json.dump(vars(opt), opt_file)

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

                    subset_ind = np.random.randint(0, len(training_data), size=(1, 1000))
                    # NOTE: removed .tolist() from subset_ind[0] as apparently not necessary
                    training_data = torch.utils.data.Subset(training_data, subset_ind[0])

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

                # set compression dictionary to validate compression_type input
                # FIXME: move this to somewhere else, not a great implementation
                comp = dict()
                # FIXME: should move this part to the opts definition at the top when it's moved to main.py
                comp['active'] = ['qat, fp']
                # does kd belong in here?
                comp['passive'] = ['ptq', 'kd']

                # NOTE: this may not work for distillation
                if opt.compress:
                    compression_scheduler = distiller.CompressionScheduler(model)
                    if opt.compression_type != 'kd':
                        compression_scheduler = distiller.file_config(model, optimizer, opt.compression_file, compression_scheduler)
                    # par, flo = model_info(model, opt)
                    # print('Before Compression:')
                    # print('Trainiable Parameters:', par)
                    # print('FLOPs:', flo)
                else:
                    compression_scheduler = None

                # NOTE: knowledge distillation - load pre-trained teacher model

                opt.kd_policy = None

                if opt.compression_type == 'kd':
                    #FIXME: add this to opts

                    # generate model using teacher args
                    teacher, parameters = generate_model(opt, teacher=True)
                    print('loading checkpoint {}'.format(opt.t_path))
                    checkpoint = torch.load(opt.t_path)
                    assert opt.t_arch == checkpoint['arch']
                    # best_prec1 = checkpoint['best_prec1']
                    # opt.begin_epoch = checkpoint['epoch']
                    teacher.load_state_dict(checkpoint['state_dict'])

                    # create a policy and add to scheduler
                    dlw = distiller.DistillationLossWeights(opt.kd_distill_wt, opt.kd_student_wt, opt.kd_teacher_wt)
                    opt.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, opt.kd_temp, dlw)
                    # compression_scheduler.add_policy(opt.kd_policy, starting_epoch=opt.kd_start_epoch,
                    #                                  ending_epoch=opt.n_epochs, frequency=1)
                    # FIXME: kd_start_epoch not defined
                    # FIXME: had to add +1 to end epoch
                    # print('begin epoch:', opt.begin_epoch)
                    end_epoch = opt.begin_epoch + opt.n_epochs
                    compression_scheduler.add_policy(opt.kd_policy, starting_epoch=opt.begin_epoch,
                                                     ending_epoch=end_epoch, frequency=1)

                # NOTE: don't want a resume path for student model - could change this?
                if opt.resume_path and opt.compression_type != 'kd':

                    print('loading checkpoint {}'.format(opt.resume_path))
                    checkpoint = torch.load(opt.resume_path)
                    assert opt.arch == checkpoint['arch']
                    best_prec1 = checkpoint['best_prec1']
                    opt.begin_epoch = checkpoint['epoch']
                    model.load_state_dict(checkpoint['state_dict'])


                print('run')
                for i in range(opt.begin_epoch, opt.n_epochs + 1):

                    if opt.compression_type in comp['active'] and opt.compress:
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

                    if opt.compression_type in comp['active'] and opt.compress:
                        compression_scheduler.on_epoch_end(i)

                # print flops and params to see if it has been reduced
                # if opt.compress:
                #     par, flo = model_info(model, opt)
                #     print('Before Compression:')
                #     print('Trainiable Parameters:', par)
                #     print('FLOPs:', flo)

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

                    # sample a defined portion of the testing dataset
                    subset_ind = np.random.randint(0, len(test_data), size=(1, 400))
                    test_data = torch.utils.data.Subset(test_data, subset_ind[0])

                    test_loader = torch.utils.data.DataLoader(
                        test_data,
                        batch_size=16,
                        shuffle=False,
                        num_workers=opt.n_threads,
                        pin_memory=True)
                    test.test(test_loader, model, opt, test_data.class_names)



                from utils.eval_ucf101 import UCFclassification
                from utils.eval_kinetics import KINETICSclassification

                opt = parse_opts()
                if opt.root_path != '':
                    opt.result_path = os.path.join(opt.root_path, opt.result_path)
                    # NOTE: below used in main.py, shouldn't be needed?
                    # opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)

                model_files = [f for f in os.listdir(opt.result_path) if os.path.splitext(f)[-1].lower() == '.pth']
                new_path = []

                f = model_files[0].split('.')[0].split('_')
                # compression type
                new_path.append(f[2])
                # date string
                new_path.append(f[4])
                path = os.path.join(opt.result_path, 'val.json')

                # DEBUG: should subset='testing' instead? - not calculating for validation set
                if opt.dataset == 'ucf101':
                    ucf_classification = UCFclassification(opt.annotation_path, path, subset='validation', top_k=1)
                    ucf_classification.evaluate()
                    a = ucf_classification.hit_at_k

                    ucf_classification = UCFclassification(opt.annotation_path, path, subset='validation', top_k=5)
                    ucf_classification.evaluate()
                    b = ucf_classification.hit_at_k

                elif dataset == 'kinetics':
                    kinetics_classification = KINETICSclassification(opt.annotation_path, path, subset='validation', top_k=1, check_status=False)
                    kinetics_classification.evaluate()
                    a = kinetics_classification.hit_at_k

                    kinetics_classification = KINETICSclassification(opt.annotation_path, path, subset='validation', top_k=5, check_status=False)
                    kinetics_classification.evaluate()
                    b = kinetics_classification.hit_at_k


                with open(path, "w") as text_file:
                    text_file.write("Parameters: \n")
                    text_file.write("Temperature: {}, Distill Weight {}, Student Weight {} \n".format(opt.kd_temp, opt.kd_distill_wt, opt.kd_student_wt))
                    text_file.write("Top-1 Accuracy: %s \n" % a)
                    text_file.write("Top-5 Accuracy: %s \n" % b)

# # use os.rename(oldfullpath, newfullpath) to move a file
# model_files = [f for f in os.listdir(opt.result_path) if os.path.isfile(os.path.join(opt.result_path, f))]
#
# # NOTE: add an 'if folder exists then add a _0 or _1 etc' loop
#
# # test if compression/date directories exists
# # FIXME: assumes compression directory already exists but want to change this!
# if not os.path.exists(os.path.join(opt.result_path, new_path[0])):
#     os.mkdir(os.path.join(opt.result_path, new_path[0]))
# if not os.path.exists(os.path.join(opt.result_path, *new_path)):
#     os.mkdir(os.path.join(opt.result_path, *new_path))
#
# # move all files
# for f in model_files:
#     # ignore hidden files
#     if not f.startswith('.'):
#         os.rename(os.path.join(opt.result_path, f), os.path.join(opt.result_path, *new_path, f))
