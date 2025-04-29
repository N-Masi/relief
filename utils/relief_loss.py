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

    TODO: figure out how to integrate all-channel averaged loss vs specific vars in output dict
    Example of what the returned dictionary may look like:
    {
        'country/USA': 0.5,
        'country/MEX': 0.7,
        ...
        'UNSDG-subregion/Northern America': 0.2,
        ...
        'landcover/land': 0.3,
        'landcover/ocean': 0.2,
        'landcover/lake': 0.25,
        ...
        'worldBankIncomeGroup/high-income': 0.6,
        'worldBankIncomeGroup/low-income': 0.8,
        ...
        'population/high-density': 0.65,
        'population/low-density': 0.4
    }

    Currently supports 1440x721 resolution (0.25degree) predictions.
    '''
    if not pred.shape == ground_truth.shape:
        raise ValueError('Predictions and ground truth values must have the same dimensions.')
    if not len(pred.shape) == 3:
        raise ValueError('Tensors must be 3-dimensional. Make sure variables with \
            multiple vertical levels have a channel for each level.')
    if channels_to_measure:
        for key in channels_to_measure.keys():
            if not type(key) == int:
                raise ValueError('')

    # TODO: support different resolutions, with data from WB2 and/or regridding.
    assert sorted(pred.shape)[1] == 721
    assert sorted(pred.shape)[2] == 1440


