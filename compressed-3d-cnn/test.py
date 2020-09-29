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


def test_eval(data_loader, model, criterion, opt, logger=None):

    model.eval()

    loss_ls = []
    top1_ls = []
    top5_ls = []

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        targets = targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        outputs = model(inputs)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))

        loss_ls.append(loss.item())
        top1_ls.append(prec1.item())
        top5_ls.append(prec5.item())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    return sum(top1_ls)/len(top1_ls), sum(top5_ls)/len(top5_ls), sum(loss_ls)/len(loss_ls)
