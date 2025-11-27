#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals

# External modules
from abc import ABC, abstractmethod

import numpy as np


class DustAttenuationModel(ABC):
    """
    Abstract class for the dust attenuation laws
    """

    def __init__(self):
        pass

    @abstractmethod
    def compute_observed_flux(self, flux_intrinsic):
        """
        Abstract method to compute observed flux after applying extinction law

        Parameters
        ----------
        flux_intrinsic

        Returns
        -------

        """
        pass

    @abstractmethod
    def get_dust_attenuation_per_wavelength(self):
        """
        Abstract method to compute the attenuation as a function of wavelength

        Returns
        -------

        """
        pass


class Calzetti2000DustAttenuationModel(DustAttenuationModel):
    def __init__(self, wavelength_grid):
        """
        This class implements the dust attenuation law presented in Calzetti et al. 2000.

        Parameters
        ----------
        wavelength_grid: numpy.array
            Array of wavelengths in units of Å, shape=(n_wavelengths,)
        """
        super().__init__()
        self.wavelength_grid = wavelength_grid
        self.Es_B_V = None
        self.A_lambda = None

    def compute_observed_flux(self, flux_intrinsic):
        """
        This function applies the dust attenuation law to the source intrinsic flux density and returns
        the source observed flux density, formula (2) in Calzetti et al. 2000

        Parameters
        ----------
        flux_intrinsic: numpy.array
            Array of intrinsic flux densities, shape=(n_wavelengths,)

        Returns
        -------
        flux_observed: numpy.array
            Array of observed flux densities, shape=(n_wavelengths,)
        """
        flux_observed = flux_intrinsic * 10 ** (-0.4 * self.A_lambda * self.Es_B_V)

        return flux_observed

    def get_color_excess_stellar_continuum(self, E_B_V):
        """
        This function computes the color excess of the stellar continuum E_s(B-V) from the color excess derived from
        the nebular gas emission lines E(B-V), formula (3) in Calzetti et al. 2000

        Parameters
        ----------
        E_B_V: float
            color excess derived from nebular gas emission lines

        Returns
        -------
        Es_B_V: float
            color excess of the stellar continuum

        """
        self.Es_B_V = 0.44 * E_B_V

        return self.Es_B_V

    def get_dust_attenuation_per_wavelength(self):
        """
        This function returns the full wavelength-dependent part of the Calzetti et al. 2000 dust attenuation curve,
        formula (4) in Calzetti et al. 2000

        Returns
        -------
        A_lambda: numpy.array
            Array of dust attenuation as a function of wavelength, shape=(n_wavelengths,)

        """
        A_lambda_1200_6300_AA = self.get_dust_attenuation_1200_6300_AA()
        A_lambda_6300_22000_AA = self.get_dust_attenuation_6300_22000_AA()
        self.A_lambda = np.concatenate([A_lambda_1200_6300_AA, A_lambda_6300_22000_AA])

        return self.A_lambda

    def get_dust_attenuation_1200_6300_AA(self):
        """
        This function returns the 1200-6300 Å range of the Calzetti et al. 2000  dust attenuation curve,
        formula (4) in Calzetti et al. 2000

        Returns
        -------
        A_lambda_1200_6300_AA: numpy.array
             Array of dust attenuation as a function of wavelength in the 1200-6300 Å range, shape=(m_wavelengths,),
             with m_wavelengths <= n_wavelenghts

        """
        mask_lambda_1200_6300_AA = np.where((self.wavelength_grid >= 1200) & (self.wavelength_grid < 6300))
        A_lambda_1200_6300_AA = (
            2.659
            * (
                -2.156
                + (1.509 / self.wavelength_grid[mask_lambda_1200_6300_AA])
                - (0.198 / self.wavelength_grid[mask_lambda_1200_6300_AA] ** 2)
                + (0.011 / self.wavelength_grid[mask_lambda_1200_6300_AA] ** 3)
            )
            + 4.05
        )

        return A_lambda_1200_6300_AA

    def get_dust_attenuation_6300_22000_AA(self):
        """
        This function returns the 6300-22000 Å range of the Calzetti et al. 2000  dust attenuation curve,
        formula (4) in Calzetti et al. 2000

        Returns
        -------
        A_lambda_6300_22000_AA: numpy.array
             Array of dust attenuation as a function of wavelength in the 6300-22000 Å range, shape=(l_wavelengths,),
             with l_wavelengths <= n_wavelenghts

        """
        mask_lambda_6300_22000_AA = np.where((self.wavelength_grid >= 6300) & (self.wavelength_grid <= 22000))
        A_lambda_6300_22000_AA = (
            2.659 * (-1.857 + (1.040 / self.wavelength_grid[mask_lambda_6300_22000_AA])) + 4.05
        )

        return A_lambda_6300_22000_AA


class KriekConroy2013DustAttenuationModel(DustAttenuationModel):
    def __init__(self, wavelength_grid, A_V, n, lambda_V=5500.0):
        """
        This class implements the dust attenuation law presented in Kriek and Conroy 2013

        Parameters
        ----------
        wavelength_grid: numpy.array
            Array of wavelengths in units of Å, shape=(n_wavelengths,)
        A_V: float
            Dust attenuation in the V-band, corresponds to `dust2' parameter in FSPS
        n: float
            Power-law slope modifier of the Calzetti et al. 2000 law, corresponds to `dust_index' parameter in FSPS
        lambda_V: float
            V-band central wavelength, default value is 5500 Å
        """
        super().__init__()
        self.wavelength_grid = wavelength_grid
        self.A_V = A_V
        self.lambda_V = lambda_V
        self.n = n
        self.A_lambda = None

    def compute_observed_flux(self, flux_intrinsic):
        """
        This function applies the dust attenuation law to the source intrinsic flux density and returns
        the source observed flux density

        Parameters
        ----------
        flux_intrinsic: numpy.array
            Array of intrinsic flux densities, shape=(n_wavelengths,)

        Returns
        -------
        flux_observed: numpy.array
            Array of observed flux densities, shape=(n_wavelengths,)
        """
        flux_observed = flux_intrinsic * 10 ** (-0.4 * self.A_lambda)

        return flux_observed

    def get_dust_attenuation_per_wavelength(self, fwhm_bump=350.0, dust_bump_wavelength=2175.0):
        """
        This function returns dust attenuation curve as a function of wavelength,
        formula (1) in Kriek and Conroy 2013

        Parameters
        ----------
        fwhm_bump: float
            Full width at half-maximum of the UV bump, defaul value is 350 Å
        dust_bump_wavelength: float
            Central wavelength of the UV dust bump, default value is 2175 Å

        Returns
        -------
        A_lambda: numpy.array
            Array of dust attenuation as a function of wavelength, shape=(n_wavelengths,)

        """
        calzetti_attenuation_law = Calzetti2000DustAttenuationModel(self.wavelength_grid)
        A_lambda_Calzetti = calzetti_attenuation_law.get_dust_attenuation_per_wavelength()
        D_lambda = self.get_UV_bump(fwhm_bump=fwhm_bump, dust_bump_wavelength=dust_bump_wavelength)
        self.A_lambda = (
            (self.A_V / 4.05)
            * (A_lambda_Calzetti * D_lambda)
            * (self.wavelength_grid / self.lambda_V) ** self.n
        )

        return self.A_lambda

    def compute_bump_amplitude(self):
        """
        This function computes the UV bump amplitude, formula (3) in Kriek and Conroy 2013

        Returns
        -------
        E_b: float
            Amplitude of the UV dust bump

        """
        E_b = 0.85 - 1.9 * self.n

        return E_b

    def get_UV_bump(self, fwhm_bump=350.0, dust_bump_wavelength=2175.0):
        """
        This function computes the Lorentzian-like Drude profile used to parametrize the UV bump,
        formula (2) in Kriek and Conroy 2013

        Parameters
        ----------
        fwhm_bump: float
            Full width at half-maximum of the UV bump, defaul value is 350 Å
        dust_bump_wavelength: float
            Central wavelength of the UV dust bump, default value is 2175 Å

        Returns
        -------
        D_lambda: numpy.array
            UV bump Lorentzian-like Drude profile as a function of wavelength, shape=(n_wavelengths,)

        """
        E_b = self.compute_bump_amplitude()
        D_lambda = (E_b * (self.wavelength_grid * fwhm_bump) ** 2) / (
            (self.wavelength_grid**2 - dust_bump_wavelength**2) ** 2 + (self.wavelength_grid * fwhm_bump) ** 2
        )

        return D_lambda


class BirthCloudLeja2017DustAttenuationModel(DustAttenuationModel):
    def __init__(self, wavelength_grid, A_birth_cloud, lambda_V=5500.0):
        """
        This class implements the birth-cloud component of the dust attenuation used in Leja et al. 2013.
        This component attenuates nebular emission and stellar emission only from stars formed in the last 10 Myr,
        the typical timescale for the disruption of a molecular cloud, formula (7) in Leja et al. 2017

        Parameters
        ----------
        wavelength_grid: numpy.array
            Array of wavelengths in units of Å, shape=(n_wavelengths,)
        A_birth_cloud: float
            Birth cloud attenuation value, `dust1' parameter in FSPS
        lambda_V: float
            V-band central wavelength, default value is 5500 Å
        """
        super().__init__()
        self.wavelength_grid = wavelength_grid
        self.A_birth_cloud = A_birth_cloud
        self.lambda_V = lambda_V
        self.A_lambda = None

    def compute_observed_flux(self, flux_intrinsic):
        """
        This function applies the dust attenuation law to the source intrinsic flux density and returns
        the source observed flux density

        Parameters
        ----------
        flux_intrinsic: numpy.array
            Array of intrinsic flux densities, shape=(n_wavelengths,)

        Returns
        -------
        flux_observed: numpy.array
            Array of observed flux densities, shape=(n_wavelengths,)
        """
        flux_observed = flux_intrinsic * 10 ** (-0.4 * self.A_lambda)

        return flux_observed

    def get_dust_attenuation_per_wavelength(self):
        """
        This function returns dust attenuation curve as a function of wavelength,
        formula (7) in Leja et al. 2017

        Returns
        -------
        A_lambda: numpy.array
            Array of dust attenuation as a function of wavelength, shape=(n_wavelengths,)

        """

        self.A_lambda = self.A_birth_cloud * (self.wavelength_grid / self.lambda_V) ** (-1.0)

        return self.A_lambda
