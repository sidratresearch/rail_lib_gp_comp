#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals

# External modules
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import truncnorm


class StellarMetallicityModel(ABC):
    """
    Abstract class for the stellar metallicity model.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_stellar_metallicities(self):
        """
        Abstract method to get stellar metallicities.

        Returns
        -------

        """
        pass


class Gallazzi2005StellarMetallicityModel(StellarMetallicityModel):
    def __init__(self, stellar_masses):
        """
        The stellar metallicity in Leja+17,19 is modelled as a truncated normal distribution with mean and standard
        deviation taken from the Gallazzi et al. 2005 mass-metallicity relation (see table 2 in Gallazzi+2005),
        with limits min=-1.98, max=0.19. The standard deviation is taken as the 84th - 16th percentile range from the
        Gallazzi+2005 z=0 relation (see table 1 in Leja+19).

        Parameters
        ----------
        stellar_masses: numpy.array
            stellar masses in units of log(M*/M*_solar)
        """
        super().__init__()
        self.stellar_masses = stellar_masses
        self.stellar_mass_bin_edges = np.array(
            [
                8.91,
                9.11,
                9.31,
                9.51,
                9.72,
                9.91,
                10.11,
                10.31,
                10.51,
                10.72,
                10.91,
                11.11,
                11.31,
                11.51,
                11.72,
                11.91,
            ]
        )
        self.stellar_metallicity_means = np.array(
            [
                -0.60,
                -0.61,
                -0.65,
                -0.61,
                -0.52,
                -0.41,
                -0.23,
                -0.11,
                -0.01,
                0.04,
                0.07,
                0.10,
                0.12,
                0.13,
                0.14,
                0.15,
            ]
        )
        self.stellar_metallicity_16thperc = np.array(
            [
                -1.11,
                -1.07,
                -1.10,
                -1.03,
                -0.97,
                -0.90,
                -0.80,
                -0.65,
                -0.41,
                -0.24,
                -0.14,
                -0.09,
                -0.06,
                -0.04,
                -0.03,
                -0.03,
            ]
        )
        self.stellar_metallicity_84thperc = np.array(
            [
                -0.00,
                -0.00,
                -0.05,
                -0.01,
                0.05,
                0.09,
                0.14,
                0.17,
                0.20,
                0.22,
                0.24,
                0.25,
                0.26,
                0.28,
                0.29,
                0.30,
            ]
        )
        self.stellar_metallicity_stds = np.abs(
            self.stellar_metallicity_84thperc - self.stellar_metallicity_16thperc
        )

    def get_stellar_metallicities(self):
        """
        This function draws values of the stellar metallicity from the Gallazzi+2005 relation modelled as a
        truncated normal distribution. Either you interpolate across the Gallazzi+2005 relation or you fix the
        stellar mass at closest value to the stellar mass bin edges.

        Returns
        -------
        log_Z_over_Zsolar: numpy.array
            stellar metallicities in units of solar metallicity drawn from the truncated normal distribution
        """
        log_Z_over_Zsolar = np.empty_like(self.stellar_masses)
        for i in range(len(self.stellar_masses)):
            idx_min = (np.abs(self.stellar_mass_bin_edges - self.stellar_masses[i])).argmin()
            a, b = (-1.98 - self.stellar_metallicity_means[idx_min]) / self.stellar_metallicity_stds[
                idx_min
            ], (0.19 - self.stellar_metallicity_means[idx_min]) / self.stellar_metallicity_stds[idx_min]
            log_Z_over_Zsolar[i] = truncnorm.rvs(
                a,
                b,
                loc=self.stellar_metallicity_means[idx_min],
                scale=self.stellar_metallicity_stds[idx_min],
                size=1,
            )

        return log_Z_over_Zsolar
