import os
import sys
import logging
import numpy as np
import pandas as pd
import distiller

from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from util import *

class Pruner():

    def __init__(self, model, net_params):

        self.net_params = net_params
        self.model = model


    def init_pruning(model, net_params):
        """perform pruning before last_layer fine-tuning on a given pre-trained model
        """

        for param_name, sparsity in net_params:
            if model.state_dict()[param_name].dim() not in [2, 4, 5]:
                continue

            sparsity = float(sparsity)
            sparsity_level = {param_name: sparsity}
            pruner = distiller.pruning.SparsityLevelParameterPruner(name="sensitivity", levels=sparsity_level)
            policy = distiller.PruningPolicy(pruner, pruner_args=None)
            scheduler = distiller.CompressionScheduler(model)
            scheduler.add_policy(policy, epochs=[0])
            # compute the pruning mask per the pruner and apply the mask on the weights
            scheduler.on_epoch_begin(0)
            scheduler.mask_all_weights()

        return model


    def get_params(opt):
        """get network parameter names and desired sparsity level from model summary/sensitivity analysis
        """

        with open(opt.summary_path, newline='') as f:
            df = pd.read_csv(f)
            # only prune Conv3d layers at the moment
            df = df[df['Type']=='Conv3d']

        params = []

        for i in df.index:
            params.append((df['Name'][i], df['Sparsity'][i]))

        return params


    def get_names(opt):
        """get network parameter names and desired sparsity level from model summary/sensitivity analysis
        """

        with open(opt.summary_path, newline='') as f:
            df = pd.read_csv(f)
            # only prune Conv3d layers at the moment
            df = df[df['Type']=='Conv3d']

        params = []

        for i in df.index:
            params.append(df['Name'][i])

        return params
