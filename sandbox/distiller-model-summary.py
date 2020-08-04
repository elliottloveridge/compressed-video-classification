import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from distiller.model_summaries import *
from distiller.models import create_model
from distiller.apputils import *
import torch
import torchvision
import qgrid

# Load some common jupyter code
%run distiller_jupyter_helpers.ipynb
import ipywidgets as widgets
from ipywidgets import interactive, interact, Layout

# Some models have long node names and require longer lines
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

def pretty_int(i):
    return "{:,}".format(i)

dataset = 'ucf101'
arch = 'mobilenetv2'
checkpoint_file = None

if checkpoint_file is not None:
    model = create_model(pretrained=True, dataset=dataset, arch=arch)
    load_checkpoint(model, checkpoint_file)
else:
    model = create_model(pretrained=True, dataset=dataset, arch=arch, parallel=False)

df = distiller.model_performance_summary(model, dummy_input, 1)

# You can display summaries using several backends, and each has its advantages and disadvantages, so you will want to use them in different situations.
print("Weights shapes, sizes and statistics (showing only FC and convolution layers):")
print("\tTotal IFM footprint (elements): " + "{:,}".format(df['IFM volume'].sum()))
print("\tTotal OFM footprint (elements): " + "{:,}".format(df['OFM volume'].sum()))
print("\tTotal weights footprint (elements): " + "{:,}".format(df['Weights volume'].sum()))

# 1. As a textual table
#t = distiller.model_performance_tbl_summary(model, dummy_input, 1)
#print(t)

# 2. As a plain Pandas dataframe
# display(df)

# 3. As a QGrid table, which you can sort and filter.

qgrid.show_grid(df)
