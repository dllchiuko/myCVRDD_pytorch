import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


def otsu(pcr, bins):
    threshold_t = 0
    max_g = 0

    for i in range(bins):
        t = i / bins
        n0 = pcr[np.where(pcr < t)]
        n1 = pcr[np.where(pcr >= t)]
        w0 = len(n0) / len(pcr)
        w1 = len(n1) / len(pcr)

        if w0 == 0 or w1 == 0:
            continue
        u0 = np.mean(n0) if len(n0) > 0 else 0.
        u1 = np.mean(n1) if len(n0) > 0 else 0.

        s0 = np.std(n0)
        s1 = np.std(n1)

        g = (w0 * w1 * ((u0 - u1) ** 2)) / (s0 * w0 + s1 * w1)
        if g > max_g:
            max_g = g
            threshold_t = t
    return threshold_t


def get_threshold(data, otsu_func=otsu):
    threshold = []
    for i in range(6):
        x = data[data.duration_level == i][['pcr']]
        threshold.append(otsu_func(x['pcr'].to_numpy(), 256))
    threshold = pd.DataFrame(threshold).reset_index()
    threshold.columns = ['duration_level', 'threshold']
    return threshold


def reorder_id(x, id):
    enc = OrdinalEncoder()
    x[id] = enc.fit_transform(x[id].to_numpy().reshape(-1, 1)).astype('int32')  # 任意行，1列
    return x[id]


    