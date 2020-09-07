import torch
from torch.autograd import Variable
import time
import os
import sys
import distiller

from util import *


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, compression_scheduler):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        # DEBUG: len(data_loader) may not be right size
        if compression_scheduler is not None:
            compression_scheduler.on_minibatch_begin(epoch, minibatch_id=i,
            minibatches_per_epoch=len(data_loader))

        if not opt.no_cuda:
            targets = targets.cuda()
        inputs = Variable(inputs)
        targets = Variable(targets)
        if opt.kd_policy is None:
            # Revert to a "normal" forward-prop call if no knowledge distillation policy is present
            outputs = model(inputs)
        else:
            # forward pass through 
            outputs = opt.kd_policy.forward(inputs)
        # outputs = model(inputs)
        loss = criterion(outputs, targets)

        # before backwards pass - update loss to include regularization
        if compression_scheduler is not None:
            loss = compression_scheduler.before_backward_pass(epoch, minibatch_id=i,
            minibatches_per_epoch=len(data_loader), loss=loss)

        losses.update(loss.data, inputs.size(0))
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()

        if compression_scheduler is not None:
            compression_scheduler.before_parameter_optimization(epoch, minibatch_id=i,
            minibatches_per_epoch=len(data_loader), optimizer=optimizer)

        optimizer.step()

        if compression_scheduler is not None:
            compression_scheduler.on_minibatch_end(epoch, minibatch_id=i,
            minibatches_per_epoch=len(data_loader))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
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
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

    #if epoch % opt.checkpoint == 0:
    #    save_file_path = os.path.join(opt.result_path,
    #                                  'save_{}.pth'.format(epoch))
    #    states = {
    #        'epoch': epoch + 1,
    #        'arch': opt.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #    }
    #    torch.save(states, save_file_path)
