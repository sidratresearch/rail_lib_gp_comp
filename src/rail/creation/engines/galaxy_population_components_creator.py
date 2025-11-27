#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals

# External modules
import os

import numpy as np
from ceci.config import StageParameter as Param
from diffstar.defaults import DEFAULT_N_STEPS, FB, LGT0, T_BIRTH_MIN
from diffstar.sfh import get_sfh_from_mah_kern
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
from dsps.utils import cumulative_mstar_formed
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from rail.core.data import Hdf5Handle
from rail.core.stage import RailStage
from rail.creation.engine import Creator

# RAIL modules
import rail
from rail.lib_gp_comp.utils.utils import multiInterp2


class DiffskyGalaxyPopulationCreator(Creator):
    """
    Derived class of Creator that samples galaxy properties from population parameters generated with
    DiffskyGalaxyPopulationModeler.
    """

    name = "DiffskyGalaxyPopulationCreator"
    entrypoint_function = "sample"  # the user-facing science function for this class
    config_options = RailStage.config_options.copy()

    config_options.update(
        log10_age_universe=Param(float, LGT0, msg="Base-10 log of the age of the universe in Gyr."),
        cosmic_baryon_fraction=Param(float, FB, msg="Cosmic baryon fraction."),
        t_min_table=Param(float, T_TABLE_MIN, msg="Lower value of the Universe age time grid."),
        t_max_table=Param(float, 10**LGT0, msg="Upper value of the Universe age time grid."),
        n_time_steps=Param(int, DEFAULT_N_STEPS, msg="Number of steps of the Universe age time grid."),
        tacc_integration_min=Param(
            float, T_BIRTH_MIN, msg="Earliest time to use in the tacc integrations. " "Default is 0.01 Gyr."
        ),
        cosmology_parameters=Param(
            tuple,
            DEFAULT_COSMOLOGY,
            msg="NamedTuple storing parameters of a flat w0-wa cdm " "cosmology, default is Planck15.",
        ),
        catalog_redshift_key=Param(str, "redshift", msg="Redshift keyword in the skysim/diffsky catalog."),
        catalog_metallicity_key=Param(
            str, "lg_met_mean", msg="Stellar metallicity keyword in the skysim/diffsky " "catalog."
        ),
        catalog_metallicity_scatter_key=Param(
            str, "lg_met_scatter", msg="Stellar metallicity scatter keyword in the " "skysim/diffsky catalog."
        ),
        cosmic_time_grid_key=Param(
            str, "cosmic_time_grid", msg="Cosmic time grid keyword in the output catalog."
        ),
        star_formation_history_key=Param(
            str, "star_formation_history", msg="Star-formation history keyword in the output catalog."
        ),
        star_formation_rate_key=Param(
            str, "star_formation_rate", msg="Star-formation rate keyword in the output catalog."
        ),
        stellar_mass_history_key=Param(
            str, "stellar_mass_history", msg="Stellar mass history keyword in the output catalog."
        ),
        stellar_mass_key=Param(str, "stellar_mass", msg="Stellar mass keyword in the output catalog."),
    )

    inputs = [("model", Hdf5Handle)]
    outputs = [("output", Hdf5Handle)]

    def __init__(self, args, comm=None):
        """

        Parameters
        ----------
        args:
        comm:
        """
        RailStage.__init__(self, args, comm=comm)

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
            n_steps=self.config.n_time_steps,
            tacc_integration_min=self.config.tacc_integration_min,
            tobs_loop="vmap",
            galpop_loop="vmap",
        )
        star_formation_histories = sfh_from_mah_kern(
            cosmic_time_grid,
            mah_params,
            ms_params,
            q_params,
            self.config.log10_age_universe,
            self.config.cosmic_baryon_fraction,
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
        time_of_observation = age_at_z(redshifts, *self.config.cosmology_parameters)
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
        time_of_observation = age_at_z(redshifts, *self.config.cosmology_parameters)
        log_stellar_masses = multiInterp2(time_of_observation, cosmic_time_grid, log_stellar_mass_histories)

        return log_stellar_masses

    def _sample_galaxy_properties_from_model(self, mah_params, ms_params, q_params, redshifts):
        """
        This function samples the galaxy properties from the skysim/diffsky galaxy population model.
        Redshifts are simply read from the input catalog. The star-formation
        histories, the star-formation rates, the stellar mass histories and the stellar masses are instead computed
        with diffmah and diffstar using the population parameters sampled from the model.

        Parameters
        -------
        mah_params: numpy.array
            ndarray of shape (n_gal, 4) containing the diffmah population parameters.
        ms_params: numpy.array
            ndarray of shape (n_gal, 5) containing the diffstar main sequence population parameters.
        q_params: numpy.array
            ndarray of shape (n_gal, 4) containing the diffstar quenching population parameters.
        redshifts: numpy.array
            Array of galaxy redshifts.

        Returns
        -------
        cosmic_time_grid: numpy.array
            Universe age time grid of length=n_time_steps.
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
        """

        cosmic_time_grid = np.linspace(
            self.config.t_min_table, self.config.t_max_table, self.config.n_time_steps
        )
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
            star_formation_histories,
            star_formation_rates,
            log_stellar_mass_histories,
            log_stellar_masses,
        )

    def sample(self, seed: int = None, input_data=None, **kwargs):
        r"""
        Samples galaxy properties from diffsky/skysim model and stores them into an Hdf5Handle

        Parameters
        ----------
        seed: int
            The random seed to control sampling
        input_data: Hdf5Handle
            This is the input diffsky/skysim population properties catalog path.

        Returns
        -------
        output: Hdf5Handle
            Hdf5 Handle storing the sampled galaxy properties.

        Notes
        -----
        This method puts  `seed` into the stage configuration data, which makes them available to other methods.
        It then calls the `run` method. Finally, the `Hdf5Handle` associated to the `output` tag is returned.

        """
        if input_data is None:
            RAIL_LIB_GP_COMP_DIR = os.path.abspath(
                os.path.join(os.path.dirname(rail.lib_gp_comp.__file__), "..")
            )
            default_files_folder = os.path.join(
                RAIL_LIB_GP_COMP_DIR, "examples_data", "creation_data", "data"
            )
            input_data = os.path.join(default_files_folder, "model_DiffskyGalaxyPopulationModeler.hdf5")

        self.config["seed"] = seed
        self.config.update(**kwargs)
        self.set_data("model", input_data)
        self.run()
        self.finalize()
        output = self.get_handle("output")
        return output

    def run(self):
        """
        This function computes the galaxy properties from the diffsky/skysim model using diffmah and diffstar
        functions. The sampled properties are those needed as input for rail_dsps and rail_fsps to work.
        """
        self.model = self.get_data("model")
        redshifts = self.model[self.config.catalog_redshift_key][()]
        stellar_metallicities = self.model[self.config.catalog_metallicity_key][()]
        stellar_metallicities_scatter = self.model[self.config.catalog_metallicity_scatter_key][()]
        mah_params = self.model["mah_params"][()]
        ms_params = self.model["ms_params"][()]
        q_params = self.model["q_params"][()]
        (
            cosmic_time_grid,
            star_formation_histories,
            star_formation_rates,
            log_stellar_mass_histories,
            log_stellar_masses,
        ) = self._sample_galaxy_properties_from_model(mah_params, ms_params, q_params, redshifts)

        galaxy_properties = {
            self.config.catalog_redshift_key: redshifts,
            self.config.catalog_metallicity_key: stellar_metallicities,
            self.config.catalog_metallicity_scatter_key: stellar_metallicities_scatter,
            self.config.cosmic_time_grid_key: np.full(
                (len(redshifts), len(cosmic_time_grid)), cosmic_time_grid
            ),
            self.config.star_formation_history_key: star_formation_histories,
            self.config.star_formation_rate_key: star_formation_rates,
            self.config.stellar_mass_history_key: log_stellar_mass_histories,
            self.config.stellar_mass_key: log_stellar_masses,
        }
        self.add_data("output", galaxy_properties)
