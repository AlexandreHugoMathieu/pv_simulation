# Created by A. MATHIEU at 08/05/2022
import pandas as pd
import numpy as np
import os
import pickle

from src.config import ROOT
from tqdm import tqdm


def get_weather():
    weather_df = pd.read_csv(ROOT / "data" / "weather_chambery.csv", skiprows=35, sep=";")

    # Date treatment
    weather_df["Time"].replace("24:00", "00:00")
    weather_df.index = pd.to_datetime(weather_df["# Date"]) + pd.to_timedelta(weather_df["Time"] + ':00')
    weather_df = weather_df.tz_localize('UTC').tz_convert("CET")
    weather_df = weather_df.drop(["# Date", "Time"], axis=1)

    # Conversion to W/m2 (timestep of 15mins)
    irr_cols = ["Direct Inclined", "Diffuse Inclined", "Global Inclined", "Direct Horiz", "Diffuse Horiz",
                "Global Horiz", 'Clear-Sky']
    weather_df[irr_cols] = weather_df[irr_cols] * 4

    # Temperature Conversion
    weather_df['Temperature'] = weather_df['Temperature'] - 273.15 if weather_df['Temperature'].mean() > 200 else \
        weather_df['Temperature']

    # Rename columns
    weather_df = weather_df.rename(
        columns={"Direct Inclined": "Gib_w.m2", "Diffuse Inclined": "Gid_w.m2", "Global Inclined": "Gi_w.m2",
                 "Direct Horiz": "Ghb_w.m2", "Diffuse Horiz": "Ghd_w.m2", "Global Horiz": "Gh_w.m2",
                 'Clear-Sky': "Ghc_w.m2", 'Temperature': "Ta_C", "Relative Humidity": "RH_perc",
                 'Wind speed': "wind_speed_m.s",
                 'Rainfall': "rain_mm"})
    weather_df = weather_df[
        ['Gib_w.m2', 'Gid_w.m2', 'Gi_w.m2', 'Ghb_w.m2', 'Ghd_w.m2', 'Gh_w.m2', 'Gh_w.m2',"Ghc_w.m2", 'Ta_C', 'RH_perc',
         'wind_speed_m.s', 'rain_mm']]

    # Identify nans
    weather_df = weather_df.astype(float)
    weather_df = weather_df.replace(-999, np.nan)

    # Get 20 years of data
    index_20y = pd.date_range("20000101", "20200101", freq="15min", tz="CET", inclusive="left")
    weather_df_previous = weather_df.reindex(index_20y).shift(-365 * 96 * 10 - 96 * 3).dropna()
    weather_20y = pd.concat([weather_df_previous, weather_df], axis=0)
    weather_20y = weather_20y[~weather_20y.index.duplicated(keep='first')].reindex(index_20y).ffill().bfill()

    # pm data
    pm_data = get_pm_data(date_range=pd.date_range("20000101", "20200101", freq="H", tz="CET", inclusive="left"),
                          site="Grenoble Les Frenes", store_pkl=True)
    weather_20y[["pm_2_5_g.m3", "pm_10_g.m3"]] = pm_data
    # Different Granularity (1h and 15mins)
    weather_20y[["pm_2_5_g.m3", "pm_10_g.m3"]] = weather_20y[["pm_2_5_g.m3", "pm_10_g.m3"]].ffill(limit=3).bfill(limit=3)
    weather_20y.loc[:, "Ee_w.m2"] = weather_20y.loc[:, "Gi_w.m2"]

    return weather_20y


def get_pm_data(date_range=pd.date_range("20210101", "20220101", freq="H", tz="CET", inclusive="left"),
                site: str = "Grenoble Les Frenes", store_pkl: bool=True):
    date_range_str = date_range.min().strftime("%Y_%m_%d_%H%M") + "_" + \
                     date_range.max().strftime("%Y_%m_%d_%H%M") + "_" + \
                     date_range.freqstr
    path_store = str(ROOT / "data" / f"pm_data_{site.replace(' ', '_')}_{date_range_str}.pkl")
    if store_pkl and os.path.exists(path_store):
        with open(path_store, "rb") as input_file:
            pm_data = pickle.load(input_file)

    else:
        pm_data = pd.DataFrame(index=date_range, columns=["pm_2_5_g.m3", "pm_10_g.m3"])

        for date in tqdm(np.unique(date_range.date)):
            year = date.strftime("%Y")
            date_str = date.strftime("%Y-%m-%d")
            url = f"https://files.data.gouv.fr/lcsqa/concentrations-de-polluants-atmospheriques-reglementes/temps-reel/" \
                  f"{year}/FR_E2_{date_str}.csv"
            raw_data = pd.read_csv(url, sep=";")
            raw_data_site = raw_data[(raw_data["nom site"] == site)]

            data_10 = raw_data_site[(raw_data_site["Polluant"] == "PM10")].set_index("Date de début")["valeur"]
            data_25 = raw_data_site[(raw_data_site["Polluant"] == "PM2.5")].set_index("Date de début")["valeur"]
            data_10.index = pd.to_datetime(data_10.index).tz_localize("CET", ambiguous=True,
                                                                      nonexistent='shift_forward')
            data_25.index = pd.to_datetime(data_25.index).tz_localize("CET", ambiguous=True,
                                                                      nonexistent='shift_forward')
            pm_data.loc[data_10.index, "pm_10_g.m3"] = data_10 / 1000 / 1000  # conversion en g/m3
            pm_data.loc[data_25.index, "pm_2_5_g.m3"] = data_25 / 1000 / 1000  # conversion en g/m3

        pm_data = pm_data.ffill()

        if store_pkl and os.path.exists(str(ROOT / "data")):
            with open(path_store, 'wb') as handle:
                pickle.dump(pm_data, handle)

    return pm_data


def pm_data_extrapolation_20y():
    pm_data = get_pm_data()
    date_range = pd.date_range("20000101", "20200101", freq="H", tz="CET", inclusive="left")
    pm_data_20y = pd.DataFrame(index=date_range, columns=pm_data.columns)

    for month in tqdm(np.unique(pm_data.index.month)):
        pm_data_month = pm_data.loc[pm_data.index.month == month]
        for day in np.unique(pm_data_month.index.day):
            pm_data_day = pm_data_month.loc[pm_data_month.index.day == day]
            for hour in np.unique(pm_data_month.index.hour):
                if not pm_data_day.loc[pm_data_day.index.hour == hour].empty:
                    pm_data_hour = pm_data_day.loc[pm_data_day.index.hour == hour].iloc[0]
                    pm_data_20y.loc[
                        (pm_data_20y.index.month == month) & (pm_data_20y.index.day == day) & (
                                pm_data_20y.index.hour == hour)] = pm_data_hour.values

    date_range_str = date_range.min().strftime("%Y_%m_%d_%H%M") + "_" + \
                     date_range.max().strftime("%Y_%m_%d_%H%M") + "_" + \
                     date_range.freqstr
    site = "Grenoble Les Frenes"
    path_store = str(ROOT / "data" / f"pm_data_{site.replace(' ', '_')}_{date_range_str}.pkl")
    with open(path_store, 'wb') as handle:
        pickle.dump(pm_data_20y, handle)

    return None

def get_insitu_weather():
    weather_df = pd.read_excel(ROOT / "data" / "perf_bipv_meteo_7.xlsx")
