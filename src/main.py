import pandas as pd
import pvlib
import os
from pathlib import Path

from src.config import ROOT

from src.weather.weather_main import get_weather
from src.simulation.simulation_generation import simulation_ui

if __name__ == "__main__":
    # Configuration  parameters
    pv_params = pvlib.pvsystem.retrieve_sam('cecmod')['Canadian_Solar_Inc__CS5P_220M']
    pv_params["tilt"] = 40
    pv_params["strings"] = 2
    pv_params["modules_serie"] = 12
    pv_params["depo_veloc"] = {'2_5': 0.02, '10': 0.04}
    pv_params["cleaning_threshold"] = 5

    # Get weather    # Assume effective irradiance = in-plane irradiance
    weather_df = get_weather().iloc[:96 * 365]

     # Electrical Model with failure scenarios
    filename = (Path(os.getcwd()) / "data" / f"simu_{pd.Timestamp('now').strftime('%Y%m%d_%Hh%M')}.pkl")
    data_UI = simulation_ui(weather_df, pv_params, pkl=True, filename=filename)
