#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals

# External modules
from abc import ABC, abstractmethod

import numpy as np


class DustEmissionModel(ABC):
    """
    Abstract class for the dust emission model.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_minimum_radiation_field_strength(self):
        """
        Abstract method to get the minimum radiation field strength that heats the dust.

        Parameters
        ----------

        Returns
        -------

        """
        pass

    @abstractmethod
    def get_fraction_grain_mass_pah(self):
        """
        Abstract method to get the fraction of grain mass in PAHs.

        Returns
        -------

        """
        pass

    @abstractmethod
    def get_relative_contribution_dust_heated(self):
        """
        Abstract method to get the relative contribution of dust heated  at a radiation field strength of Umin.

        Returns
        -------

        """
        pass


class DustEmissionProspectorAlphaModel(DustEmissionModel):
    def __init__(self, number_samples):
        """
        This class returns the dust emission model parameters needed for the Prospector Alpha model (Leja+17,19).
        The parameters are used to generate dust emission with Prospector/FSPS using the Draine&Li 2007 dust
        emission model.

        Parameters
        ----------
        number_samples: int
            Number of model parameters to draw from distributions
        """
        super().__init__()
        self.n_samples = number_samples

    def get_minimum_radiation_field_strength(self):
        """
        This function draws parameters of the Draine & Li (2007) dust emission model. Specifies the minimum
        radiation field strength in units of the MW value.

        Returns
        -------
        U_min_dust: numpy.array
            minimum radiation field strengths
        """
        U_min_dust = np.random.uniform(0.1, 15, self.n_samples)

        return U_min_dust

    def get_fraction_grain_mass_pah(self):
        """
        This function draws parameters of the Draine & Li (2007) dust emission model. Specifies the grain size
        distribution through the fraction of grain mass in PAHs. This parameter has units of %.

        Returns
        -------
        Qpah_dust: numpy.array
            grain mass fractions in PAH
        """
        Qpah_dust = np.random.uniform(0.1, 7.0, self.n_samples)

        return Qpah_dust

    def get_relative_contribution_dust_heated(self):
        """
        This function draws parameters of the Draine & Li (2007) dust emission model. Specifies the relative
        contribution of dust heated at a radiation field strength of Umin and dust heated at Umin <= U <= Umax.

        Returns
        -------
        gamma_dust: numpy.array
            relative contributions of dust heated at Umin
        """
        gamma_dust = np.random.uniform(0.0, 0.15, self.n_samples)

        return gamma_dust
