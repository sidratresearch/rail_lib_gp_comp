#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals

# External modules
from abc import ABC, abstractmethod

import numpy as np
from astropy import units as u

# creation imports
from utils.utils_luminosity_function import (
    compute_lower_truncation_scaled_schechter_random_variable,
    gamma_function_integration_for_redshift,
    get_minimum_limiting_absolute_magnitude,
    sample_from_schechter_function,
)


class LuminosityFunctionModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample_galaxy_redshifts_from_luminosity_function(self, redshift_number_density, number_of_galaxies):
        pass

    @abstractmethod
    def sample_absolute_magnitudes_from_luminosity_function(self, galaxy_redshifts):
        pass

    @abstractmethod
    def get_comoving_number_density(self):
        pass

    @abstractmethod
    def get_redshift_number_density(self, comoving_number_density):
        pass

    @abstractmethod
    def get_expected_number_of_galaxies(self, redshift_number_density):
        pass

    @abstractmethod
    def get_number_of_galaxies(self, expected_number_of_galaxies):
        pass


class SchechterLuminosityFunctionModel(LuminosityFunctionModel):
    def __init__(
        self,
        input_redshift_grid,
        m_star,
        phi_star,
        alpha,
        apparent_magnitude_limit,
        sky_area,
        cosmology_object,
    ):
        """
        This class implements the Schechter functional form for the galaxy luminosity function.

        Parameters
        ----------
        input_redshift_grid: numpy.array
            input redshift grid on which the Schechter function parameters are
            evaluated (Galaxies are sampled over this redshift range)
        m_star: float or numpy.array
            characteristic absolute magnitude, it can be constant or function of  input_redshift_grid
        phi_star: float or numpy.array
            normalization density, it can be constant or function of input_redshift_grid
        alpha: float or numpy.array
            faint-end power-law slope, it can be constant or function of input_redshift_grid
        apparent_magnitude_limit: float
            limiting apparent magnitude of sampled galaxies
        sky_area: float
            sky area over which galaxies are sampled in units of square degrees
        cosmology_object: astropy.cosmology
            astropy.cosmology objects for computing distance moduli and other cosmological related quantities.
        """

        super().__init__()
        self.input_redshift_grid = input_redshift_grid
        self.m_star = m_star
        self.phi_star = phi_star
        self.alpha = alpha
        self.apparent_magnitude_limit = apparent_magnitude_limit
        self.sky_area = (
            sky_area * (np.pi / 180) ** 2 * u.sr
        )  # transforming in units of solid angle for astropy
        self.cosmology = cosmology_object

    def sample_galaxy_redshifts_from_luminosity_function(self, redshift_number_density, number_of_galaxies):
        """
        This function samples galaxy redshift from a Schechter function given a redshift number density and the
        expected number of galaxies. The functions to compute these two quantities are present in the class itself.
        The function first uses the cumulative trapezoidal rule to get the redshift cumulative distribution function
        (cdf) and then samples galaxy redshifts.

        Parameters
        ----------
        redshift_number_density: numpy.array
            redshift number densities for each redshift in grid
        number_of_galaxies: int
            Poisson realization of the expected number of galaxies

        Returns
        -------
        sampled_galaxy_redshifts: numpy.array
            sampled galaxy redshifts from Schechter luminosity function

        """

        cdf = redshift_number_density
        np.cumsum(
            (redshift_number_density[1:] + redshift_number_density[:-1])
            / 2
            * np.diff(self.input_redshift_grid),
            out=cdf[1:],
        )
        cdf[0] = 0
        cdf /= cdf[-1]
        sampled_galaxy_redshifts = np.interp(
            np.random.rand(number_of_galaxies), cdf, self.input_redshift_grid
        )

        return sampled_galaxy_redshifts

    def sample_absolute_magnitudes_from_luminosity_function(
        self,
        sampled_galaxy_redshifts,
        maximum_limiting_absolute_magnitude=1e3,
        size=None,
        scale=1.0,
        resolution=1000,
    ):
        """
        This function samples galaxy absolute magnitudes from the Schechter function given the sampled galaxy
        redshifts, size of the sampling and maximum absolute magnitude.

        Parameters
        ----------
        sampled_galaxy_redshifts: numpy.array
            sampled galaxy redshifts from Schechter luminosity function
        maximum_limiting_absolute_magnitude: numpy.array
            maximum limiting absolute magnitude for each galaxy redshift
        size: int
            output shape of samples. If size is None and scale is a scalar, a
            single sample is returned. If size is None and scale is an array, an
            array of samples is returned with the same shape as scale. Default is None
        scale: float or numpy.array
            scale factor for the returned samples. Default is 1
        resolution: int
            resolution of the inverse transform sampling spline

        Returns
        -------
        sampled_absolute_magnitudes: numpy.array
            absolute magnitudes sampled from a Schechter luminosity function for each input galaxy redshift

        """

        x_min = get_minimum_limiting_absolute_magnitude(
            sampled_galaxy_redshifts, self.apparent_magnitude_limit, self.cosmology, self.m_star
        )
        sampled_absolute_magnitudes = sample_from_schechter_function(
            self.alpha,
            x_min,
            maximum_limiting_absolute_magnitude,
            size=size,
            scale=scale,
            resolution=resolution,
        )
        np.log10(sampled_absolute_magnitudes, out=sampled_absolute_magnitudes)
        sampled_absolute_magnitudes *= -2.5
        sampled_absolute_magnitudes += self.m_star

        return sampled_absolute_magnitudes

    def get_comoving_number_density(self):
        """
        This function computes the comoving number density of galaxies as the normalisation phi_star times incomplete
        gamma.

        Returns
        -------
        comoving_number_density: numpy.array
            comoving number densities of galaxies for each redshift in grid.

        """

        lnxmin = compute_lower_truncation_scaled_schechter_random_variable(
            self.input_redshift_grid, self.apparent_magnitude_limit, self.cosmology, self.m_star
        )

        gamma = gamma_function_integration_for_redshift(lnxmin, self.alpha)

        comoving_number_density = self.phi_star * gamma

        return comoving_number_density

    def get_redshift_number_density(self, comoving_number_density):
        """
        This function computes the redshift number density dN/dz provided a cosmology to compute the differential
        comoving volume at each redshift in the grid, a sky area and the comoving number density.

        Parameters
        ----------
        comoving_number_density: numpy.array
            comoving number densities of galaxies for each redshift in grid

        Returns
        -------
        redshift_number_density: numpy.array
            redshift number densities for each redshift in grid

        """

        redshift_number_density = (
            self.cosmology.differential_comoving_volume(self.input_redshift_grid) * self.sky_area
        ).to_value("Mpc3")
        redshift_number_density *= comoving_number_density

        return redshift_number_density

    def get_expected_number_of_galaxies(self, redshift_number_density):
        """
        This function computes the expected number of galaxies by integrating the redshift number density
        over the redshift grid.

        Parameters
        ----------
        redshift_number_density: numpy.array
            redshift number densities for each redshift in grid

        Returns
        -------
        expected_number_of_galaxies: int
            expected number of galaxies

        """

        expected_number_of_galaxies = np.trapz(redshift_number_density, self.input_redshift_grid)

        return expected_number_of_galaxies

    def get_number_of_galaxies(self, expected_number_of_galaxies):
        """

        Parameters
        ----------
        expected_number_of_galaxies: int
            expected number of galaxies

        Returns
        -------
        number_of_galaxies: int
            Poisson realization of the expected number of galaxies.

        """

        number_of_galaxies = np.random.poisson(expected_number_of_galaxies)
        return number_of_galaxies
