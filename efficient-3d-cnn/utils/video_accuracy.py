from eval_ucf101 import UCFclassification
from eval_kinetics import KINETICSclassification

opt = parse_opts()
if opt.root_path != '':
    opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
    opt.result_path = os.path.join(opt.root_path, opt.result_path)

print(opt)
# FIXME: add an opt for top_k = 1 or 5 so you can set it!

if opt.dataset == 'ucf101':

    # check if you need str() for paths
    ucf_classification = UCFclassification(opt.annotation__path, opt.result_path, subset='validation', top_k=1)
    ucf_classification.evaluate()
    print(ucf_classification.hit_at_k)

elif opt.dataset == 'kinetics':
    kinetics_classification = KINETICSclassification('../annotation_Kinetics/kinetics.json',
                                           '../results/val.json',
                                           subset='validation',
                                           top_k=1,
                                           check_status=False)
    kinetics_classification.evaluate()
    print(kinetics_classification.hit_at_k)
