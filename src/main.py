import pandas as pd
import pvlib

from config import ROOT
from src.weather.weather_main import get_weather
from src.models.temp_model import temp_NOTC
from src.models.elec_model import get_vi

# Configuration  parameters
pv_params = pvlib.pvsystem.retrieve_sam('cecmod')['Canadian_Solar_Inc__CS5P_220M']
pv_params["tilt"] = 40
pv_params["strings"] = 2
pv_params["modules_serie"] = 12
pv_params["depo_veloc"] = {'2_5': 0.02, '10': 0.04}
pv_params["cleaning_threshold"] = 5

# Get weather
weather_df = get_weather()

# Assume effective irradiance = in-plane irradiance
weather_df.loc[:, "Ee_w.m2"] = weather_df.loc[:, "Gi_w.m2"]

# Temperature Model
weather_df.loc[:, "Tc_C"] = temp_NOTC(weather_df, pv_params)

# Electrical Model with failure scenarios
data_UI = get_vi(weather_df, pv_params, method="singlediode_regress")



# Failure Scenario
data_UI.to_pickle(str(ROOT / "data" / f"data_UI.pkl"))
weather_df.to_pickle(str(ROOT / "data" / f"weather_df.pkl"))

data_UI = pd.read_pickle(str(ROOT / "data" / f"data_UI.pkl"))
weather_df = pd.read_pickle(str(ROOT / "data" / f"weather_df.pkl"))