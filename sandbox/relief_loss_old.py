 import torch
import numpy as np
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.surface_area import *

def relief_loss(
    pred: torch.Tensor | np.ndarray, 
    ground_truth: torch.Tensor | np.ndarray,
    weigh_by_surface_area: bool = True,
    channels_to_measure: Optional[dict] = None) -> dict:
    '''
    Takes in a prediction of climate variables as well as the ground truth 
    for those variables at the same time step and returns a dictionary from 
    ReLIEF metadata group to average loss.

    Assumes that predictions and ground truth values are already normalized
    (i.e., the data fed into the model to generate preds was normalized).

    Shape of data must be 3-dimensional. That is, variables with different
    vertical levels must be reorganized so there is one channel per variable
    per level.

    By default, loss is averaged (with weighting by surface area, this can be
    toggled off) across all channels. The <channels_to_measure> 

    TODO: figure out how to include all-channel averaged loss vs specific vars in output dict
    TODO: also have a subdirectory for different lead times
    Example of what the returned dictionary may look like:
    

    Currently supports 721x1440 resolution (0.25degree gridline-registered) predictions.
    '''
    if not pred.shape == ground_truth.shape:
        raise ValueError('Predictions and ground truth values must have the same dimensions.')
    if not len(pred.shape) == 3:
        raise ValueError('Tensors must be 3-dimensional. Make sure variables with \
            multiple vertical levels have a channel for each level.')
    if channels_to_measure:
        for key in channels_to_measure.keys():
            if not type(key) == int:
                raise ValueError('') # TODO

    # TODO: support different resolutions, with data from WB2 and/or regridding.
    assert sorted(pred.shape)[1] == 721
    assert sorted(pred.shape)[2] == 1440





def relief_loss(preds, ground_truth, channel_names, loss_fn):
    assert preds.shape == ground_truth.shape

    relief_loss = {}

    # TODO: for each channel
    
    # TODO: 


    # TODO: handle reducing loss across channels

def markdown_relief_loss(relief_loss, filename):
    # TODO: make a .md file that breaks down the relief loss
