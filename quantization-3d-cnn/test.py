import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json

from util import AverageMeter

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

# pruning sensitivity analysis required a single function to test/evaluate and return losses, so will not dump values to json
def test_eval(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    # time not required for this method
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):

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

        # print every 100th sample - not using as low sample size
        # if i % 100 == 0:
        print('[{}/{}]\t'.format(i + 1, len(data_loader)))

    with open(os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),'w') as f:
        json.dump(test_results, f)

    print('eval')

    # FIXME: don't want to hardcode val.json anywhere as could be test.json
    opt.result_path = os.path.join(opt.result_path, 'val.json')

    # FIXME: need to return all accuracy values rather than an average
    if opt.dataset == 'ucf101':
        # FIXME: need to change the subset to 'testing' rather than 'validation' - needs to match the above test_subset!
        ucf_classification = UCFclassification(opt.annotation_path, opt.result_path, subset='validation', top_k=1, full_eval=True)
        ucf_classification.evaluate()
        top1 = ucf_classification.hit_at_k

        ucf_classification = UCFclassification(opt.annotation_path, opt.result_path, subset='validation', top_k=5, full_eval=True)
        ucf_classification.evaluate()
        top5 = ucf_classification.hit_at_k

    # now need to return losses and top-1/5 accuracy
    # NOTE: are test_results the same as losses?

    print(test_results)
    print()
    print()
    print(top1)

    return test_results, top1, top5
