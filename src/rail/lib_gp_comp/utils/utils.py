#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals

# External modules
import numpy as np


def multiInterp2(x, xp, fp):
    i = np.arange(x.size)
    j = np.searchsorted(xp, x) - 1
    d = (x - xp[j]) / (xp[j + 1] - xp[j])
    return (1 - d) * fp[i, j] + fp[i, j + 1] * d
