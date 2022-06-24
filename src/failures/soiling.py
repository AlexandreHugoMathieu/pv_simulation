# Created by A. MATHIEU at 09/05/2022
import pandas as pd

from pvlib.soiling import hsu


def hsu_soiling(weather_df: pd.DataFrame, pv_params: dict, rain_accum_period=pd.Timedelta("24h")):
    tilt = pv_params["tilt"]
    depo_veloc = pv_params["depo_veloc"] if "depo_veloc" in list(pv_params.keys()) else {'2_5': 0.0009, '10': 0.004}
    cleaning_threshold = pv_params["cleaning_threshold"] if "cleaning_threshold" in list(pv_params.keys()) else 5

    rainfall = weather_df["rain_mm"]
    pm2_5 = weather_df["pm_2_5_g.m3"]
    pm10 = weather_df["pm_10_g.m3"]

    s_ratio = hsu(rainfall, cleaning_threshold, tilt, pm2_5, pm10, depo_veloc, rain_accum_period=rain_accum_period)
    return s_ratio
