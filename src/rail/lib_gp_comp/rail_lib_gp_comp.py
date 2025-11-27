#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals


def check_implemented_components(input_component):
    """

    Parameters
    ----------
    input_component

    Returns
    -------

    """
    implemented_modelling_components = ["Schechter"]

    if input_component in implemented_modelling_components:
        return True
    else:
        return False
