#! /usr/bin/env python

# Copyright (C) 2023 Luca Tortorelli, LSST DESC PZ WG
# Author: Luca Tortorelli

# System imports
from __future__ import absolute_import, division, print_function, unicode_literals

# External modules
import os

import numpy as np
from ceci.config import StageParameter as Param
from rail.core.data import Hdf5Handle, TableHandle
from rail.core.stage import RailStage

# RAIL modules
from rail.creation.engine import Modeler

import rail


class DiffskyGalaxyPopulationModeler(Modeler):
    r"""
    Derived class of Modeler for creating a mock galaxy population using diffsky/skysim library.
    This class in particular samples the population parameters from the input diffsky/skysim catalog.
    """

    name = "DiffskyGalaxyPopulationModeler"
    entrypoint_function = "fit_model"  # the user-facing science function for this class
    config_options = RailStage.config_options.copy()

    config_options.update(
        diffmah_keys=Param(
            list,
            ["diffmah_logmp_fit", "diffmah_mah_logtc", "diffmah_early_index", "diffmah_late_index"],
            msg="Keywords list in the skysim/diffsky catalog that store " "diffmah parameters.",
        ),
        diffstar_ms_keys=Param(
            list,
            [
                "diffstar_u_lgmcrit",
                "diffstar_u_lgy_at_mcrit",
                "diffstar_u_indx_lo",
                "diffstar_u_indx_hi",
                "diffstar_u_tau_dep",
            ],
            msg="Keywords list in the skysim/diffsky catalog that store diffstar"
            " main sequence parameters.",
        ),
        diffstar_q_keys=Param(
            list,
            ["diffstar_u_qt", "diffstar_u_qs", "diffstar_u_q_drop", "diffstar_u_q_rejuv"],
            msg="Keywords list in the skysim/diffsky catalog that store diffstar" " quenching parameters.",
        ),
        catalog_redshift_key=Param(str, "redshift", msg="Redshift keyword in the skysim/diffsky catalog."),
        catalog_metallicity_key=Param(
            str, "lg_met_mean", msg="Stellar metallicity keyword in the skysim/diffsky " "catalog."
        ),
        catalog_metallicity_scatter_key=Param(
            str, "lg_met_scatter", msg="Stellar metallicity scatter keyword in the " "skysim/diffsky catalog."
        ),
    )

    inputs = [("input", TableHandle)]
    outputs = [("model", Hdf5Handle)]

    def __init__(self, args, comm=None):
        """
        This function initializes the DiffskyGalaxyPopulationModeler class. It checks whether the input
        diffsky/skysim catalog exists. If not, it is downloaded from the NERSC public directory.

        Parameters
        ----------
        args:
        comm:

        """

        RailStage.__init__(self, args, comm=comm)

    def _get_fit_params(self, data):
        """
        Read the mock galaxy diffsky/skysim table and return the diffmah and diffstar fit params.

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
        mah_params = np.array([data[key] for key in self.config.diffmah_keys]).T
        ms_params = np.array([data[key] for key in self.config.diffstar_ms_keys]).T
        q_params = np.array([data[key] for key in self.config.diffstar_q_keys]).T

        return mah_params, ms_params, q_params

    def fit_model(self, input_data=None):
        """
        This function samples the population parameters from the diffsky/skysim galaxy population model and stores
        them into an Hdf5Handle.

        Parameters
        ----------
        input_data: str
            This is the input diffsky/skysim catalog path.

        Returns
        -------
        model: Hdf5Handle
            Hdf5 table storing the population parameters.
        """
        if input_data is None:
            RAIL_LIB_GP_COMP_DIR = os.path.abspath(
                os.path.join(os.path.dirname(rail.lib_gp_comp.__file__), "..")
            )
            default_files_folder = os.path.join(
                RAIL_LIB_GP_COMP_DIR, "examples_data", "creation_data", "data"
            )
            input_data = os.path.join(default_files_folder, "skysim_v3.1.0_10k_lssty1cut.pq")
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        model = self.get_handle("model")
        return model

    def run(self):
        """
        Run method. It Calls `_get_fit_params` to sample the population parameters.
        """
        input_skysim_properties = self.get_data("input")
        mah_params, ms_params, q_params = self._get_fit_params(input_skysim_properties)
        redshifts = input_skysim_properties[self.config.catalog_redshift_key]
        stellar_metallicities = input_skysim_properties[self.config.catalog_metallicity_key]
        stellar_metallicities_scatter = input_skysim_properties[self.config.catalog_metallicity_scatter_key]
        population_parameters = {
            "mah_params": mah_params,
            "ms_params": ms_params,
            "q_params": q_params,
            self.config.catalog_redshift_key: redshifts.values,
            self.config.catalog_metallicity_key: stellar_metallicities.values,
            self.config.catalog_metallicity_scatter_key: stellar_metallicities_scatter.values,
        }
        self.add_data("model", population_parameters)
