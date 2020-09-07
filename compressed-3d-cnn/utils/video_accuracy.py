import argparse
import os

from eval_ucf101 import UCFclassification
from eval_kinetics import KINETICSclassification


def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/data', type=str, help='Root directory path of data')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--annotation_path', default='/app/quantization-3d-cnn/annotation_UCF101/ucf101_01.json', type=str, help='Annotation file path')
    parser.add_argument('--dataset', default='ucf101', type=str, help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    args = parser.parse_args()

    return args


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

# save results as a text file
path = os.path.join(opt.result_path, 'accuracy.txt')
with open(path, "w") as text_file:
    text_file.write("Top-1 Accuracy: %s /n" % a)
    text_file.write("Top-5 Accuracy: %s" % b)

# use os.rename(oldfullpath, newfullpath) to move a file
model_files = [f for f in os.listdir(opt.result_path) if os.path.isfile(os.path.join(opt.result_path, f))]

# test if compression/date directories exists
# FIXME: assumes compression directory already exists but want to change this!
if not os.path.exists(os.path.join(opt.result_path, new_path[0])):
    os.mkdir(os.path.join(opt.result_path, new_path[0]))
if not os.path.exists(os.path.join(opt.result_path, *new_path)):
    os.mkdir(os.path.join(opt.result_path, *new_path))

# move all files
for f in model_files:
    # ignore hidden files
    if not f.startswith('.'):
        os.rename(os.path.join(opt.result_path, f), os.path.join(opt.result_path, *new_path, f))
