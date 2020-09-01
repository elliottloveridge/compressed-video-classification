import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json

from util import *

from torch import nn
from torch import optim
from torch.optim import lr_scheduler

# DEBUG: need to get this working
from utils.eval_ucf101 import UCFclassification


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[int(locs[i])],
            'score': float(sorted_scores[i])
        })

    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            inputs = Variable(inputs)
        outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs, dim=1)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]

        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}.json'.format(
                        opt.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)


# # pruning sensitivity analysis required a single function to test/evaluate
# def test_eval(data_loader, model, opt, class_names, criterion):
#
#     # NOTE: added from distiller.app_utils.image_classifier
#     losses = {'objective_loss': AverageMeter()}
#
#     print('test')
#     model.eval()
#     output_buffer = []
#     previous_video_id = ''
#     test_results = {'results': {}}
#     for i, (inputs, targets) in enumerate(data_loader):
#         with torch.no_grad():
#             inputs = Variable(inputs)
#             targets = Variable(torch.tensor(targets.values))
#         outputs = model(inputs)
#
#         if not opt.no_softmax_in_test:
#             outputs = F.softmax(outputs, dim=1)
#
#         # FIXME: do I need this loss? or could I use test_results score?
#         print('output:', outputs.size)
#         print('target:', targets.size)
#         print()
#         print(outputs)
#         loss = criterion(outputs, targets)
#         losses['objective_loss'].add(loss.item())
#
#         for j in range(outputs.size(0)):
#             if not (i == 0 and j == 0) and targets[j] != previous_video_id:
#                 calculate_video_results(output_buffer, previous_video_id,
#                                         test_results, class_names)
#                 output_buffer = []
#             output_buffer.append(outputs[j].data.cpu())
#             previous_video_id = targets[j]
#
#         # FIXME: this currently outputs [1/7] etc - why so low?
#         # batches of 16 with dataset of 100 currently 100/16 = 6.25 ~ 7
#         print('[{}/{}]\t'.format(i + 1, len(data_loader)))
#
#     with open(os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),'w') as f:
#         json.dump(test_results, f)
#
#     print('eval')
#
#     #  FIXME: could just call the eval function here?
#     # NOTE: full_eval=True returns all accuracy values rather than an average
#     if opt.dataset == 'ucf101':
#         # FIXME: should subset be changed to 'testing' from 'validation'?
#         ucf_classification = UCFclassification(opt.annotation_path,
#         os.path.join(opt.result_path, 'val.json'), subset='validation',
#         top_k=1, full_eval=True)
#         ucf_classification.evaluate()
#         top1 = ucf_classification.hit_at_k
#
#         ucf_classification = UCFclassification(opt.annotation_path,
#         opt.result_path, subset='validation', top_k=5, full_eval=True)
#         ucf_classification.evaluate()
#         top5 = ucf_classification.hit_at_k
#
#     print('top1:', top1)
#     print('top5:', top5)
#
#     print('losses:')
#     print(losses['objective_loss'])
#     # now need to return losses and top-1/5 accuracy
#     # FIXME: are test_results the same as losses? - nope!
#
#
#     return losses['objective_loss'], top1, top5

def test_eval(data_loader, model, criterion, opt, logger=None):

    model.eval()

    loss_ls = []
    top1_ls = []
    top5_ls = []

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        # FIXME: fix the below
        # if not opt.no_cuda:
        targets = targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        outputs = model(inputs)

        criterion = nn.CrossEntropyLoss()
        # if not opt.no_cuda:
        criterion = criterion.cuda()

        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        # top1.update(prec1, inputs.size(0))
        # top5.update(prec5, inputs.size(0))

        print(loss.item())

        print(prec1.item())
        print(prec5.item())

        # losses.update(losses.data(), inputs.size(0))

        loss_ls.append(loss.item())
        top1_ls.append(prec1.item())
        top5_ls.append(prec5.item())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # print('Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
        #       'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #       'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
        #       'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
        #           i + 1,
        #           len(data_loader),
        #           batch_time=batch_time,
        #           data_time=data_time,
        #           loss=losses,
        #           top1=top1,
        #           top5=top5))

    # logger.log({'loss': losses.avg.item(),
    #             'prec1': top1.avg.item(),
    #             'prec5': top5.avg.item()})

    print()
    print('results')
    print(loss_ls, top1_ls, top5_ls)

    # return losses.count.item(), prec1.count.item(), prec5.count.item()
