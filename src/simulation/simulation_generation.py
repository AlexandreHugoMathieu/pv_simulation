# Created by A. MATHIEU at 24/06/2022
import os

import pandas as pd

from typing import Callable
from pathlib import Path

from src.models.temp_model import temp_NOTC
from src.models.elec_model import get_vi


def simulation_ui(weather_df: pd.DataFrame,
                  pv_params: dict,
                  temp_model: Callable = temp_NOTC,
                  elec_model: Callable = get_vi,
                  params_elec: dict = {"method": "singlediode"},
                  pkl: bool = True,
                  filename=(Path(
                      os.getcwd()) / f"simu_{pd.Timestamp('now').strftime('%Y%m%d_%Hh%M')}.pkl")) -> pd.DataFrame:
    """
    Generates UI according to weather, configuration and chosen models

    :param weather_df: weather dataframe containing irradiance, temperature etc...
    :param pv_params: configuration paramaters
    :param temp_model: temperature model
    :param elec_model: electrical model to generate UI
    :param params_elec: electrical model parameters
    :param pkl: save into pickle File ?
    :param filename: which name to save the pickle file in  ?

    :return: dataframe with u, i and p.
    """
    if os.path.exists(filename):
        data_UI= pd.read_pickle(filename)

    else:
        # Temp and elec simulation
        weather_df.loc[:, "Tc_C"] = temp_model(weather_df, pv_params)
        data_UI = elec_model(weather_df, pv_params, **params_elec)

    if pkl:
        data_UI.to_pickle(filename)

    return data_UI
