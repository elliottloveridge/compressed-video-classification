import torch
from torch import nn

from models import c3d, squeezenet, mobilenet, shufflenet, mobilenetv2, shufflenetv2, resnext, resnet, csn


def generate_model(opt, teacher=False):
    assert opt.model in ['c3d', 'squeezenet', 'mobilenet', 'resnext', 'resnet',
                         'shufflenet', 'mobilenetv2', 'shufflenetv2', 'csn']

    if teacher:
        assert opt.t_model in ['c3d', 'squeezenet', 'mobilenet', 'resnext', 'resnet',
                               'shufflenet', 'mobilenetv2', 'shufflenetv2', 'csn']

    # NOTE: naive teacher model implementation
    if not teacher:
        model_name = opt.model
        num_classes = opt.n_classes
        sample_size = opt.sample_size
        sample_duration = opt.sample_duration
        version = opt.version
        groups = opt.groups
        width_mult = opt.width_mult
        resnet_shortcut = opt.resnet_shortcut
        resnext_cardinality = opt.resnext_cardinality
        model_depth = opt.model_depth
        model_type=opt.model_type
    else:
        model_name = opt.t_model
        num_classes = opt.t_n_classes
        sample_size = opt.t_sample_size
        sample_duration = opt.t_sample_duration
        version = opt.t_version
        groups = opt.t_groups
        width_mult = opt.t_width_mult
        resnet_shortcut = opt.t_resnet_shortcut
        resnext_cardinality = opt.t_resnext_cardinality
        model_depth = opt.t_model_depth
        # NOTE: have not included model type for CSN as no plans to use it as a teacher model
        # this may give an undefined warning!

    if model_name == 'c3d':
        from models.c3d import get_fine_tuning_parameters
        model = c3d.get_model(
            num_classes=num_classes,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_name == 'squeezenet':
        from models.squeezenet import get_fine_tuning_parameters
        model = squeezenet.get_model(
            version=version,
            num_classes=num_classes,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_name == 'shufflenet':
        from models.shufflenet import get_fine_tuning_parameters
        model = shufflenet.get_model(
            groups=groups,
            width_mult=width_mult,
            num_classes=num_classes)
    elif model_name == 'shufflenetv2':
        from models.shufflenetv2 import get_fine_tuning_parameters
        model = shufflenetv2.get_model(
            num_classes=num_classes,
            sample_size=sample_size,
            width_mult=width_mult)
    elif model_name == 'mobilenet':
        from models.mobilenet import get_fine_tuning_parameters
        model = mobilenet.get_model(
            num_classes=num_classes,
            sample_size=sample_size,
            width_mult=width_mult)
    elif model_name == 'mobilenetv2':
        from models.mobilenetv2 import get_fine_tuning_parameters
        model = mobilenetv2.get_model(
            num_classes=num_classes,
            sample_size=sample_size,
            width_mult=width_mult)
    elif model_name == 'resnext':
        assert model_depth in [50, 101, 152]
        from models.resnext import get_fine_tuning_parameters
        if model_depth == 50:
            model = resnext.resnext50(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                cardinality=resnext_cardinality,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 101:
            model = resnext.resnext101(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                cardinality=resnext_cardinality,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 152:
            model = resnext.resnext152(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                cardinality=resnext_cardinality,
                sample_size=sample_size,
                sample_duration=sample_duration)
    elif model_name == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]
        from models.resnet import get_fine_tuning_parameters
        if model_depth == 10:
            model = resnet.resnet10(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 18:
            model = resnet.resnet18(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 34:
            model = resnet.resnet34(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 50:
            model = resnet.resnet50(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 101:
            model = resnet.resnet101(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 152:
            model = resnet.resnet152(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 200:
            model = resnet.resnet200(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
    elif model_name == 'csn':
        from models.csn import get_fine_tuning_parameters
        if model_depth == 26:
            model = csn.csn26(
                num_classes=num_classes,
                sample_size=sample_size,
                width_mult=width_mult,
                model_type=model_type)
        elif model_depth == 50:
            model = csn.csn50(
                num_classes=num_classes,
                sample_size=sample_size,
                width_mult=width_mult,
                model_type=model_type)
        elif model_depth == 101:
            model = csn.csn101(
                num_classes=num_classes,
                sample_size=sample_size,
                width_mult=width_mult,
                model_type=model_type)
        elif model_depth == 152:
            model = csn.csn152(
                num_classes=num_classes,
                sample_size=sample_size,
                width_mult=width_mult,
                model_type=model_type)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

        # NOTE: this may not work when using a teacher model or csn model
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2', 'csn']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes))
                model.module.classifier = model.module.classifier.cuda()
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
                model.module.classifier = model.module.classifier.cuda()
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2', 'csn']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes)
                                )
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()
