#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals

# External modules
from abc import ABC, abstractmethod

import numpy as np


class GasEmissionModel(ABC):
    """
    Abstract class for the gas emission model.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_gas_ionization_parameter(self):
        """
        Abstract method to get the gas ionization parameter U.

        Parameters
        ----------

        Returns
        -------

        """
        pass

    @abstractmethod
    def get_gas_phase_metallicity(self):
        """
        Abstract method to get the gas-phase metallicity.

        Returns
        -------

        """
        pass


class GasEmissionProspectorAlphaModel(GasEmissionModel):
    def __init__(self, number_samples):
        """
        This class returns the gas emission model parameters needed for the Prospector Alpha model (Leja+17,19).
        The parameters are used to generate the gas emission with Prospector/FSPS using the Cloudy models
        from Nell Byler.

        Parameters
        ----------
        number_samples: int
            Number of model parameters to draw from distributions
        """
        super().__init__()
        self.n_samples = number_samples

    def get_gas_ionization_parameter(self):
        """
        This function draws values for the log of the gas ionization parameter U. In Leja+17, the value is
        kept fixed at -2, in Leja+19 to -1 to account for stronger star-forming galaxies at high redshift.

        Returns
        -------
        log_U: numpy.array
            log of the gas ionization parameter U
        """
        log_U = np.full(self.n_samples, -1, dtype=int)

        return log_U

    def get_gas_phase_metallicity(self):
        """
        This function draws values for the log of the gas-phase metallicity. Contrary to what is prescribed in the
        FSPS manual, the gas-phase metallicity is not equal to the stellar metallicity.

        Returns
        -------
        log_Zgas_over_Zsolar: numpy.array
            gas-phase metallicity in units of solar metallicity
        """
        log_Zgas_over_Zsolar = np.random.uniform(-2, 0.5, self.n_samples)

        return log_Zgas_over_Zsolar
