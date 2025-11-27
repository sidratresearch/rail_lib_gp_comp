#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals

# External modules
from abc import ABC, abstractmethod

from scipy.stats import loguniform


class AGNModel(ABC):
    """
    Abstract class for the AGN emission model
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_fraction_bolometric_luminosity_AGN(self):
        """
        Abstract method to get the AGN fractional contribution to the galaxy bolometric luminosity

        Parameters
        ----------

        Returns
        -------

        """
        pass

    @abstractmethod
    def get_optical_depth_dust_torus(self):
        """
        Abstract method to get the optical depth of the AGN dust torus

        Returns
        -------

        """
        pass


class AGNProspectorAlphaModel(AGNModel):
    def __init__(self, number_samples):
        """
        This class returns the AGN model parameters needed for the Prospector Alpha model (Leja+17,19).
        The parameters are used to generate AGN fluxes with Prospector/FSPS using the templates in Nenkova+08.

        Parameters
        ----------
        number_samples: int
            Number of model parameters to draw from distributions
        """
        super().__init__()
        self.n_samples = number_samples

    def get_fraction_bolometric_luminosity_AGN(self):
        """
        This function samples the fractional contributions of the AGN to the bolometric galaxy luminosity from a
        log-uniform distribution with min=10^{-5} and max=3 (see table 1 in Leja+19)

        Returns
        -------
        f_bol_agn: numpy.array
            agn fractional contributions to galaxy bolometric luminosities
        """
        f_bol_agn = loguniform.rvs(10 ** (-5), 3, size=self.n_samples)

        return f_bol_agn

    def get_optical_depth_dust_torus(self):
        """
        This function samples the optical depths of the AGN dust torus from a log-uniform distribution
        with min=5 and max=150 (see table 1 in Leja+19)

        Returns
        -------
        tau_agn: numpy.array
            optical depths of AGN dust torii
        """
        tau_agn = loguniform.rvs(5, 150, size=self.n_samples)

        return tau_agn
