# Created by A. MATHIEU at 23/06/2022
# This script evaluates the error per agregation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pvlib
import scipy.stats

from typing import Union

from src.config import ROOT
from src.weather.weather_main import get_insitu_weather, get_sat_weather
from src.simulation.simulation_generation import simulation_ui
from src.utils.helio_fmt import setup_helio_plt


def SE_aggregation(start: pd.Timestamp,
                   end: pd.Timestamp,
                   variable_measured: pd.DataFrame,
                   freqs: list = ["5min", "15min", "1h"]):
    """Aggregate the timeseries according to SolarEdge method"""

    series = {}
    for freq in freqs:
        ts_tmp = variable_measured.copy()
        index_freq = pd.DatetimeIndex(pd.date_range(start, end, freq=freq))
        idx = pd.DatetimeIndex(sorted(ts_tmp.index.append(index_freq)))
        idx = idx[~idx.duplicated()]
        ts_tmp = ts_tmp.reindex(idx).ffill()
        series[freq] = ts_tmp.resample(freq).mean()

    return series


def weighted_aggregation(start: pd.Timestamp,
                         end: pd.Timestamp,
                         freq_var: pd.Timedelta,
                         variable_measured: pd.DataFrame,
                         freqs: list = ["1min", "5min", "15min", "1h"]):
    """Aggregate the timeseries according to the weighted-average method"""
    index_base = pd.DatetimeIndex(pd.date_range(start, end, freq=freq_var))
    series = {}
    for freq in freqs:
        series[freq] = variable_measured.reindex(index_base).ffill().resample(freq).mean()

    return series


def apply_invervals(variable: pd.Series,
                    freq_var: pd.Timedelta,
                    measure_intervals: pd.Series):
    """
    Apply intervals on simulated measures to reflect the fact that measures are taken at random times

    :param variable: variable to be aggregated (power for instance)
    :param freq_var: frequency of the simulated variable
    :param measure_intervals: intervals between measures (in mins)

    :return: simulated variable timestamped at random intervals
    """
    measure_intervals = (measure_intervals * pd.Timedelta("1min") / freq_var).astype(int)
    measure_intervals = measure_intervals[measure_intervals.cumsum() < len(data_UI)]
    measure_intervals = measure_intervals[measure_intervals.cumsum() < len(variable)]
    variable_measured = variable.copy().iloc[measure_intervals.cumsum()]
    variable_measured = variable_measured[~variable_measured.index.duplicated()].copy()

    return variable_measured


def visualize_SE_aggreg(variable: pd.Series,
                        freq_var: pd.Timedelta,
                        measure_intervals: pd.Series,
                        freq: Union[str, pd.Timedelta]):
    variable_measured = apply_invervals(variable, freq_var, measure_intervals)
    start = variable_measured.index.min()
    end = variable_measured.index.max()

    # Aggregation
    SE_serie = SE_aggregation(start, end, variable_measured, [freq])[freq]

    # Figures
    setup_helio_plt()
    fig_freq = plt.figure(figsize=(12, 6))
    plt.title(f"Aggregation at frequency: {freq}")
    variable_measured.plot(linewidth=0, marker="o", label="Measured points")
    variable.plot(label="Simulated variable")
    SE_serie.reindex(variable.index).ffill().plot(label=f"SE-aggregation at frequency: {freq}")
    plt.legend()

    fig_raw = plt.figure(figsize=(12, 6))
    plt.title(f"Raw data")
    variable_measured.plot(linewidth=0, marker="o", label="Measured points")
    variable.plot(label="Simulated variable")
    SE_serie.reindex(variable.index).ffill().plot(label=f"SE-aggregation at frequency: {freq}")
    plt.legend()

    return fig_freq, fig_raw


def aggregation_error(variable: pd.Series,
                      freq_var: pd.Timedelta,
                      measure_intervals: pd.Series,
                      freqs: list = ["5min", "15min", "30min", "45min", "1h"]) -> pd.DataFrame:
    """
    Calculate aggregation error from SE method and weighted average method based on simulation method.

    :param variable: variable to be aggregated (power for instance)
    :param freq_var: frequency of the simulated variable
    :param measure_intervals: intervals between measures (in mins)
    :param freqs: freq to assess

    :return: DataFrame with MAE metrics
    """

    # Apply the interval distribution to the theoretical calculated power
    variable_measured = apply_invervals(variable, freq_var, measure_intervals)

    # Aggregation
    SE_series = SE_aggregation(start, end, variable_measured, freqs)
    weighted_series = weighted_aggregation(start, end, freq_var, variable_measured, ["1min"] + freqs)

    # Error calculation
    df = pd.DataFrame(columns=["SE_agg_error", "weighted_agg_error"])
    for freq in freqs:
        real_agg = variable.fillna(0).resample(freq).mean()
        error_SE = (SE_series[freq] - real_agg).fillna(0)
        error_w = (weighted_series[freq] - real_agg).fillna(0)

        # MEAN
        df.loc[freq, "SE_agg_error"] = error_SE.abs().mean() / variable.mean() * 100
        df.loc[freq, "weighted_agg_error"] = error_w.abs().mean() / variable.mean() * 100

    return df


if __name__ == "__main__":
    # Configuration  parameters
    pv_params = pvlib.pvsystem.retrieve_sam('cecmod')['Canadian_Solar_Inc__CS5P_220M']
    pv_params["tilt"] = 40
    pv_params["strings"] = 1
    pv_params["modules_serie"] = 1
    pv_params["depo_veloc"] = {'2_5': 0.02, '10': 0.04}
    pv_params["cleaning_threshold"] = 5

    # Get weather    # Assume effective irradiance = in-plane irradiance
    weather_df = get_insitu_weather()

    # Electrical Model with failure scenarios
    filename = ROOT / "data" / f"simu_SE_analysis.pkl"
    data_UI = simulation_ui(weather_df, pv_params, pkl=True, filename=filename)

    # Get aggregation errors
    variable = data_UI["Pmpp_w"].fillna(0)
    start = data_UI.index.min()
    end = data_UI.index.max()
    freq_var = pd.Timedelta(variable.index.freq)
    freqs = ["5min", "15min", "1h"]
    dist = getattr(scipy.stats, "dweibull")
    measure_intervals = pd.Series(dist.rvs(size=len(data_UI), loc=5.02, scale=2.45, c=1.22))  # min
    errors = aggregation_error(variable, freq_var, measure_intervals)
    print("Average")
    print(errors)

    # Get aggregation when clearness is low
    weather_sat = get_sat_weather()
    daily_clearness = weather_sat['Gh_w.m2'].resample("D").sum() / weather_sat['Ghc_w.m2'].resample("D").sum()
    low_clearness = daily_clearness.loc[daily_clearness < 0.50]
    variable.loc[~np.isin(data_UI.index.date, low_clearness.index.date)] = 0
    errors_clearness = aggregation_error(variable, freq_var, measure_intervals)

    print("Low clearness")
    print(errors_clearness)
