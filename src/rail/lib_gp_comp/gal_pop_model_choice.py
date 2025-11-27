#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os

# External modules
from abc import ABC, abstractmethod

import h5py
import numpy as np
import pandas as pd
from diffstar.defaults import DEFAULT_N_STEPS, FB, LGT0, T_BIRTH_MIN
from diffstar.sfh import get_sfh_from_mah_kern
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
from dsps.utils import cumulative_mstar_formed
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

# RAIL modules
from rail.core.utils import find_rail_file

from rail.lib_gp_comp.utils.utils import multiInterp2

default_files_folder = find_rail_file(os.path.join("examples_data", "creation_data", "data"))


class GalaxyPopulationModel(ABC):
    """
    Abstract class for the Galaxy Population model
    """

    def __init__(self):
        pass

    @abstractmethod
    def sample_population_parameters(self):
        pass

    @abstractmethod
    def sample_galaxy_properties(self, population_parameters):
        pass

    @abstractmethod
    def store_sampled_population_parameters(self, population_parameters):
        pass

    @abstractmethod
    def store_sampled_galaxy_properties(self, galaxy_properties):
        pass


class DiffskyGalaxyPopulationModel(GalaxyPopulationModel):
    """
    This class is an interface to the skysim v3.1.0/diffsky galaxy population model. It uses the catalogs provided
    on NERSC to sample the population properties and the galaxy physical properties.

    """

    def __init__(
        self,
        skysim_input_catalog_path=os.path.join(default_files_folder, "skysim_v3.1.0_red.csv"),
        population_parameters_table_path=os.path.join(
            default_files_folder, "skysim_v3.1.0_population_parameters.h5"
        ),
        galaxy_properties_table_path=os.path.join(default_files_folder, "skysim_v3.1.0_galaxy_properties.h5"),
        DIFFMAH_KEYS=None,
        DIFFSTAR_MS_KEYS=None,
        DIFFSTAR_Q_KEYS=None,
        log10_age_universe=LGT0,
        cosmic_baryon_fraction=FB,
        t_min_table=T_TABLE_MIN,
        t_max_table=10**LGT0,
        n_time_steps=DEFAULT_N_STEPS,
        tacc_integration_min=T_BIRTH_MIN,
        cosmology_parameters=DEFAULT_COSMOLOGY,
        catalog_redshift_key="redshift",
        catalog_metallicity_key="lg_met_mean",
        catalog_metallicity_scatter_key="lg_met_scatter",
        cosmic_time_grid_key="cosmic_time_grid",
        star_formation_history_key="star_formation_history",
        star_formation_rate_key="star_formation_rate",
        stellar_mass_history_key="stellar_mass_history",
        stellar_mass_key="stellar_mass",
    ):
        """

        Parameters
        ----------
        skysim_input_catalog_path: str
            Path to the skysim/diffsky input catalog in csv format.
        population_parameters_table_path: str
            Path to the hdf5 catalog storing the sampled population parameters.
        galaxy_properties_table_path: str
            Path to the hdf5 catalog storing the sampled galaxy properties.
        DIFFMAH_KEYS: list
            Keywords list in the skysim/diffsky catalog that store diffmah parameters.
        DIFFSTAR_MS_KEYS: list
            Keywords list in the skysim/diffsky catalog that store diffstar main sequence parameters.
        DIFFSTAR_Q_KEYS: list
            Keywords list in the skysim/diffsky catalog that store diffstar quenching parameters.
        log10_age_universe: float
            Base-10 log of the age of the universe in Gyr.
        cosmic_baryon_fraction: float
            Cosmic baryon fraction.
        t_min_table: float
            Lower value of the Universe age time grid.
        t_max_table: float
            Upper value of the Universe age time grid.
        n_time_steps: int
            Number of steps of the Universe age time grid.
        tacc_integration_min: float
            Earliest time to use in the tacc integrations. Default is 0.01 Gyr.
        cosmology_parameters: NamedTuple
            NamedTuple storing parameters of a flat w0-wa cdm cosmology, default is Planck15.
        catalog_redshift_key: str
            Redshift keyword in the skysim/diffsky catalog.
        catalog_metallicity_key: str
            Stellar metallicity keyword in the skysim/diffsky catalog.
        catalog_metallicity_scatter_key: str
            Stellar metallicity scatter keyword in the skysim/diffsky catalog.
        cosmic_time_grid_key: str
            Cosmic time grid keyword in the output catalog.
        star_formation_history_key:str
            Star-formation history keyword in the output catalog.
        star_formation_rate_key:str
            Star-formation rate keyword in the output catalog.
        stellar_mass_history_key:str
            Stellar mass history keyword in the output catalog.
        stellar_mass_key:str
            Stellar mass keyword in the output catalog.
        """
        super().__init__()

        if DIFFMAH_KEYS is None:
            self.DIFFMAH_KEYS = [
                "diffmah_logmp_fit",
                "diffmah_mah_logtc",
                "diffmah_early_index",
                "diffmah_late_index",
            ]
        if DIFFSTAR_MS_KEYS is None:
            self.DIFFSTAR_MS_KEYS = [
                "diffstar_u_" + key for key in ["lgmcrit", "lgy_at_mcrit", "indx_lo", "indx_hi", "tau_dep"]
            ]
        if DIFFSTAR_Q_KEYS is None:
            self.DIFFSTAR_Q_KEYS = ["diffstar_u_" + key for key in ["qt", "qs", "q_drop", "q_rejuv"]]

        self.log10_age_universe, self.cosmic_baryon_fraction = log10_age_universe, cosmic_baryon_fraction
        self.n_time_steps, self.tacc_integration_min = n_time_steps, tacc_integration_min
        self.cosmology_parameters, self.catalog_redshift_key = cosmology_parameters, catalog_redshift_key
        self.catalog_metallicity_key = catalog_metallicity_key
        self.catalog_metallicity_scatter_key = catalog_metallicity_scatter_key
        self.cosmic_time_grid_key, self.star_formation_history_key = (
            cosmic_time_grid_key,
            star_formation_history_key,
        )
        self.star_formation_rate_key, self.stellar_mass_history_key = (
            star_formation_rate_key,
            stellar_mass_history_key,
        )
        self.stellar_mass_key, self.t_min_table, self.t_max_table = stellar_mass_key, t_min_table, t_max_table

        self.skysim_df = pd.read_csv(skysim_input_catalog_path)
        self.population_parameters_table_path = population_parameters_table_path
        self.galaxy_properties_table_path = galaxy_properties_table_path

    def sample_population_parameters(self):
        """
        This function samples the population parameters from the skysim/diffsky input catalog that are used by
        diffmah and diffstar to generate the star-formation histories.

        Returns
        -------
        mah_params: numpy.array
            ndarray of shape (n_gal, 4) containing the diffmah population parameters.
        ms_params: numpy.array
            ndarray of shape (n_gal, 5) containing the diffstar main sequence population parameters.
        q_params: numpy.array
            ndarray of shape (n_gal, 4) containing the diffstar quenching population parameters.
        """
        mah_params, ms_params, q_params = self._get_fit_params(self.skysim_df)

        return mah_params, ms_params, q_params

    def _get_fit_params(self, data):
        """
        Read the mock galaxy skysim table and return the diffmah and diffstar fit params.

        Parameters
        ----------
        data: dataframe
            skysim/diffsky dataframe.
        Returns
        -------
        mah_params: numpy.array
            ndarray of shape (n_gal, 4) containing the diffmah population parameters.
        ms_params: numpy.array
            ndarray of shape (n_gal, 5) containing the diffstar main sequence population parameters.
        q_params: numpy.array
            ndarray of shape (n_gal, 4) containing the diffstar quenching population parameters.
        """
        mah_params = np.array([data[key] for key in self.DIFFMAH_KEYS]).T
        ms_params = np.array([data[key] for key in self.DIFFSTAR_MS_KEYS]).T
        q_params = np.array([data[key] for key in self.DIFFSTAR_Q_KEYS]).T

        return mah_params, ms_params, q_params

    def _compute_sfh_from_mah_kern(self, cosmic_time_grid, mah_params, ms_params, q_params):
        """
        JAX-jitted function that computes the star-formation histories from the diffmah kernel.

        Parameters
        ----------
        cosmic_time_grid: numpy.array
            Universe age time grid of length=n_time_steps.
        mah_params: numpy.array
            ndarray of shape (n_gal, 4) containing the diffmah population parameters.
        ms_params: numpy.array
            ndarray of shape (n_gal, 5) containing the diffstar main sequence population parameters.
        q_params: numpy.array
            ndarray of shape (n_gal, 4) containing the diffstar quenching population parameters.

        Returns
        -------
        star_formation_histories: numpy.array
            Star-formation histories ndarray of shape (n_gal, n_time_steps) containing the star-formation rates in
            Msun/yr per time bin.
        """
        sfh_from_mah_kern = get_sfh_from_mah_kern(
            n_steps=self.n_time_steps,
            tacc_integration_min=self.tacc_integration_min,
            tobs_loop="vmap",
            galpop_loop="vmap",
        )
        star_formation_histories = sfh_from_mah_kern(
            cosmic_time_grid,
            mah_params,
            ms_params,
            q_params,
            self.log10_age_universe,
            self.cosmic_baryon_fraction,
        )

        return star_formation_histories

    def _compute_sfr_from_sfh(self, cosmic_time_grid, star_formation_histories, redshifts):
        """
        This function computes the star-formation rate at the time of observation for each galaxy based on their
        computed star-formation histories.

        Parameters
        ----------
        cosmic_time_grid: numpy.array
            Universe age time grid of length=n_time_steps.
        star_formation_histories: numpy.array
            Star-formation histories ndarray of shape (n_gal, n_time_steps) containing the star-formation rates in
            Msun/yr per time bin.
        redshifts: numpy.array
            Array of galaxy redshifts.

        Returns
        -------
        star_formation_rates: numpy.array
            Array of star-formation rates in units of Msun/yr at the galaxy time of observations.
        """
        time_of_observation = age_at_z(redshifts, *self.cosmology_parameters)
        star_formation_rates = multiInterp2(time_of_observation, cosmic_time_grid, star_formation_histories)

        return star_formation_rates

    @staticmethod
    def _compute_cumulative_formed_smh(cosmic_time_grid, star_formation_histories):
        """
        This function computes the cumulative formed stellar mass histories of skysim/diffsky galaxies based
        on the computed star-formation histories.

        Parameters
        ----------
        cosmic_time_grid: numpy.array
            Universe age time grid of length=n_time_steps.
        star_formation_histories: numpy.array
            Star-formation histories ndarray of shape (n_gal, n_time_steps) containing the star-formation rates in
            Msun/yr per time bin.

        Returns
        -------
        log_stellar_mass_histories: numpy.array
            Stellar mass histories ndarray of shape (n_gal, n_time_steps) containing the cumulative formed
            stellar mass in units of log10(M*/Msun) per time bin.
        """
        _c = [None, 0]
        _calc_smh_table_vmap = jjit(vmap(cumulative_mstar_formed, in_axes=_c))
        args_smh_table = (cosmic_time_grid, star_formation_histories)
        stellar_mass_histories = _calc_smh_table_vmap(*args_smh_table)
        log_stellar_mass_histories = jnp.log10(stellar_mass_histories)

        return log_stellar_mass_histories

    def _compute_sm_from_smh(self, cosmic_time_grid, log_stellar_mass_histories, redshifts):
        """
        This function computes the log10 of the cumulative formed stellar mass at the time of observation for
        each galaxy based on their computed stellar mass histories.

        Parameters
        ----------
        cosmic_time_grid: numpy.array
            Universe age time grid of length=n_time_steps.
        log_stellar_mass_histories: numpy.array
            Stellar mass histories ndarray of shape (n_gal, n_time_steps) containing the cumulative formed
            stellar mass in units of log10(M*/Msun) per time bin.
        redshifts: numpy.array
            Array of galaxy redshifts.

        Returns
        -------
        log_stellar_masses: numpy.array
            Array of formed stellar masses in units of log10(M*/Msun) at the galaxy time of observations.
        """
        time_of_observation = age_at_z(redshifts, *self.cosmology_parameters)
        log_stellar_masses = multiInterp2(time_of_observation, cosmic_time_grid, log_stellar_mass_histories)

        return log_stellar_masses

    def sample_galaxy_properties(self, population_parameters):
        """
        This function samples the galaxy properties from the skysim/diffsky galaxy population model.
        Redshifts, stellar metallicity means and scatters are simply read from the input catalog. The star-formation
        histories, the star-formation rates, the stellar mass histories and the stellar masses are instead computed
        with diffmah and diffstar using the population parameters sampled from the model.

        Parameters
        -------
        population_parameters: tuple
            Population parameters sampled from skysim/diffsky model in the form (mah_params, ms_params, q_params).

        Returns
        -------
        cosmic_time_grid: numpy.array
            Universe age time grid of length=n_time_steps.
        redshifts: numpy.array
            Array of galaxy redshifts.
        star_formation_histories: numpy.array
            Star-formation histories ndarray of shape (n_gal, n_time_steps) containing the star-formation rates in
            Msun/yr per time bin.
        star_formation_rates: numpy.array
            Array of star-formation rates in units of Msun/yr at the galaxy time of observations.
        log_stellar_mass_histories: numpy.array
            Stellar mass histories ndarray of shape (n_gal, n_time_steps) containing the cumulative formed
            stellar mass in units of log10(M*/Msun) per time bin.
        log_stellar_masses: numpy.array
            Array of formed stellar masses in units of log10(M*/Msun) at the galaxy time of observations.
        stellar_metallicity: numpy.array
            Array of means of the stellar metallicity distribution read from skysim/diffsky.
        stellar_metallicity_scatter: numpy.array
            Array of scatters of the stellar metallicity distribution read from skysim/diffsky.
        """
        mah_params, ms_params, q_params = population_parameters
        redshifts = self.skysim_df[self.catalog_redshift_key]
        stellar_metallicity = self.skysim_df[self.catalog_metallicity_key]
        stellar_metallicity_scatter = self.skysim_df[self.catalog_metallicity_scatter_key]

        cosmic_time_grid = np.linspace(self.t_min_table, self.t_max_table, self.n_time_steps)
        star_formation_histories = self._compute_sfh_from_mah_kern(
            cosmic_time_grid, mah_params, ms_params, q_params
        )
        star_formation_rates = self._compute_sfr_from_sfh(
            cosmic_time_grid, star_formation_histories, redshifts
        )
        log_stellar_mass_histories = self._compute_cumulative_formed_smh(
            cosmic_time_grid, star_formation_histories
        )
        log_stellar_masses = self._compute_sm_from_smh(
            cosmic_time_grid, log_stellar_mass_histories, redshifts
        )

        return (
            cosmic_time_grid,
            redshifts,
            star_formation_histories,
            star_formation_rates,
            log_stellar_mass_histories,
            log_stellar_masses,
            stellar_metallicity,
            stellar_metallicity_scatter,
        )

    def store_sampled_population_parameters(self, population_parameters):
        """

        Parameters
        ----------
        population_parameters: tuple
            Population parameters sampled from skysim/diffsky model in the form (mah_params, ms_params, q_params).
        """
        mah_params, ms_params, q_params = population_parameters
        with h5py.File(self.population_parameters_table_path, "w") as h5table:
            h5table.create_dataset(name="mah_params", data=mah_params)
            h5table.create_dataset(name="ms_params", data=ms_params)
            h5table.create_dataset(name="q_params", data=q_params)

    def store_sampled_galaxy_properties(self, galaxy_properties):
        """

        Parameters
        ----------
        galaxy_properties: tuple
            Galaxy properties sampled from skysim/diffsky model in the form (cosmic_time_grid, redshifts,
            star_formation_histories, star_formation_rates, log_stellar_mass_histories,
            log_stellar_masses, stellar_metallicity, stellar_metallicity_scatter).
        """
        (
            cosmic_time_grid,
            redshifts,
            star_formation_histories,
            star_formation_rates,
            log_stellar_mass_histories,
            log_stellar_masses,
            stellar_metallicity,
            stellar_metallicity_scatter,
        ) = galaxy_properties
        with h5py.File(self.galaxy_properties_table_path, "w") as h5table:
            h5table.create_dataset(name=self.cosmic_time_grid_key, data=cosmic_time_grid)
            h5table.create_dataset(name=self.catalog_redshift_key, data=redshifts)
            h5table.create_dataset(name=self.star_formation_history_key, data=star_formation_histories)
            h5table.create_dataset(name=self.star_formation_rate_key, data=star_formation_rates)
            h5table.create_dataset(name=self.stellar_mass_history_key, data=log_stellar_mass_histories)
            h5table.create_dataset(name=self.stellar_mass_key, data=log_stellar_masses)
            h5table.create_dataset(name=self.catalog_metallicity_key, data=stellar_metallicity)
            h5table.create_dataset(
                name=self.catalog_metallicity_scatter_key, data=stellar_metallicity_scatter
            )


class ProspectorAlphaGalaxyPopulationModel(GalaxyPopulationModel):
    """ """

    def __init__(self):
        super().__init__()

    def sample_population_parameters(self):
        pass

    def sample_galaxy_properties(self, population_parameters):
        pass

    def store_sampled_population_parameters(self, population_parameters):
        pass

    def store_sampled_galaxy_properties(self, galaxy_properties):
        pass


class ProspectorBetaGalaxyPopulationModel(GalaxyPopulationModel):
    """ """

    def __init__(self):
        super().__init__()

    def sample_population_parameters(self):
        pass

    def sample_galaxy_properties(self, population_parameters):
        pass

    def store_sampled_population_parameters(self, population_parameters):
        pass

    def store_sampled_galaxy_properties(self, galaxy_properties):
        pass
