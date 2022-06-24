# Created by A. MATHIEU at 23/06/2022
# This script evaluates the error per agregation
import pandas as pd
import numpy as np
import pvlib
import scipy.stats

from src.weather import get_weather
from src.models.temp_model import temp_NOTC
from src.models import get_vi

# Configuration  parameters
pv_params = pvlib.pvsystem.retrieve_sam('cecmod')['Canadian_Solar_Inc__CS5P_220M']
pv_params["tilt"] = 40
pv_params["strings"] = 1
pv_params["modules_serie"] = 1
pv_params["depo_veloc"] = {'2_5': 0.02, '10': 0.04}
pv_params["cleaning_threshold"] = 5

# Get weather
weather_df = get_weather().iloc[:10 * 96]
weather_df = weather_df[["Gi_w.m2", "Ta_C"]].resample("10s").interpolate()

# Assume effective irradiance = in-plane irradiance
weather_df.loc[:, "Ee_w.m2"] = weather_df.loc[:, "Gi_w.m2"]

# Temperature Model
weather_df.loc[:, "Tc_C"] = temp_NOTC(weather_df, pv_params)

# Electrical Model with failure scenarios
data_UI = get_vi(weather_df, pv_params, method="singlediode", soiling=False)
data_UI.to_pickle("data_UI.pkl")

data_UI = pd.read_pickle("../data/data_UI.pkl")
daily_clearness = weather_df['Gh_w.m2'].resample("D").sum().iloc[:,0] / weather_df['Ghc_w.m2'].astype(float).resample("D").sum()
low_clearness = daily_clearness[daily_clearness < 0.50]
data_UI = data_UI[np.isin(data_UI.index.date,low_clearness.index.date)]

dist = getattr(scipy.stats, "dweibull")
df = dist.rvs(size=len(data_UI), loc=5.02,
              scale=2.45, c=1.22)  # min
df = (pd.Series(df) * 6).astype(int)  # 10s
df = df[df.cumsum() < len(data_UI)]

index_base = data_UI.index[0]

data_UI_real = data_UI.copy().iloc[df.cumsum()]

real_agg_5 = data_UI["Pmpp_w"].fillna(0).resample("5min").mean()
real_agg_15 = data_UI["Pmpp_w"].fillna(0).resample("15min").mean()
real_agg_1h = data_UI["Pmpp_w"].fillna(0).resample("1h").mean()

data_UI_real_before =data_UI_real.copy()
data_UI_real = data_UI_real[~data_UI_real.index.duplicated()].copy()

index_5min = pd.DatetimeIndex(sorted(data_UI_real.index.append(data_UI["Pmpp_w"].resample("5min").mean().index)))
index_5min = index_5min[~index_5min.duplicated()]
index_15min =  pd.DatetimeIndex(sorted(data_UI_real.index.append(data_UI["Pmpp_w"].resample("15min").mean().index)))
index_15min = index_15min[~index_15min.duplicated()]
index_1h =  pd.DatetimeIndex(sorted(data_UI_real.index.append(data_UI["Pmpp_w"].resample("1h").mean().index)))
index_1h = index_1h[~index_1h.duplicated()]

SE_agg_5 = data_UI_real["Pmpp_w"].reindex(index_5min).ffill().resample("5min").mean()
SE_agg_15 = data_UI_real["Pmpp_w"].reindex(index_15min).ffill().resample("15min").mean()
SE_agg_1h = data_UI_real["Pmpp_w"].reindex(index_1h).ffill().resample("1h").mean()
error_5 = (SE_agg_5 - real_agg_5).abs().sum() / data_UI["Pmpp_w"].fillna(0).sum() * 100
error_15 = (SE_agg_15 - real_agg_15).abs().sum() / data_UI["Pmpp_w"].fillna(0).sum() * 100
error_1h = (SE_agg_1h - real_agg_1h).abs().sum() / data_UI["Pmpp_w"].fillna(0).sum() * 100



data_UI_real = data_UI_real.fillna(0).reindex(data_UI.index).ffill()

SE_agg_5 = data_UI_real["Pmpp_w"].resample("5min").mean()
SE_agg_15 = data_UI_real["Pmpp_w"].resample("15min").mean()
SE_agg_1h = data_UI_real["Pmpp_w"].resample("1h").mean()




error_10s = (data_UI_real["Pmpp_w"].fillna(0) - data_UI["Pmpp_w"].fillna(0)).abs().sum() / data_UI["Pmpp_w"].fillna(0).sum() * 100
error_5 = (SE_agg_5 - real_agg_5).abs().sum() / data_UI["Pmpp_w"].fillna(0).sum() * 100
error_15 = (SE_agg_15 - real_agg_15).abs().sum() / data_UI["Pmpp_w"].fillna(0).sum() * 100
error_1h = (SE_agg_1h - real_agg_1h).abs().sum() / data_UI["Pmpp_w"].fillna(0).sum() * 100


print(error_5)
print(error_15)
print(error_1h)

data_UI_real_before["Pmpp_w"].plot(marker="o", label="Measured power points", color="red")
data_UI_real["Pmpp_w"].plot(label="Power with reading at different intervals", color="orange")
data_UI["Pmpp_w"].plot(label="'Real Power", color="blue")
import matplotlib.pyplot as plt
plt.legend()
