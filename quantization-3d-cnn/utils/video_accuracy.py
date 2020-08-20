import argparse
import os

from eval_ucf101 import UCFclassification
from eval_kinetics import KINETICSclassification


def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/root/data/ActivityNet', type=str, help='Root directory path of data')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--annotation_path', default='kinetics.json', type=str, help='Annotation file path')
    args = parser.parse_args()

    return args


opt = parse_opts()
if opt.root_path != '':
    opt.result_path = os.path.join(opt.root_path, opt.result_path)
    opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)

# NOTE: want to get files using os here and figure out dataset etc
ls = os.listdir(opt.result_path)
model_files = [f for f in os.listdir(opt.result_path) if os.path.splitext(f)[-1].lower() == '.pth']

for f in model_files:
    print(f)

# dataset = 'ucf101'
# result_path = '/data/results/val.json'
# annotation_path = '/app/efficient-3d-cnn/annotation_UCF101/ucf101_01.json'

if dataset == 'ucf101':

    ucf_classification = UCFclassification(annotation_path, result_path, subset='validation', top_k=1)
    ucf_classification.evaluate()
    print('top 1:', ucf_classification.hit_at_k)

    ucf_classification = UCFclassification(annotation_path, result_path, subset='validation', top_k=5)
    ucf_classification.evaluate()
    print('top 5:', ucf_classification.hit_at_k)

elif dataset == 'kinetics':
    kinetics_classification = KINETICSclassification('../annotation_Kinetics/kinetics.json',
                                           '../results/val.json',
                                           subset='validation',
                                           top_k=5,
                                           check_status=False)
    kinetics_classification.evaluate()
    print(kinetics_classification.hit_at_k)
