# Created by A. MATHIEU at 08/05/2022
import os

from pvlib import pvsystem
from scipy.interpolate import interp1d
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
from functools import partial

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import ROOT
from src.failures.soiling import hsu_soiling


def get_vi(weather_df, pv_params, method="singlediode", soiling=True):
    data = pd.DataFrame(index=weather_df.index, columns=["Pmpp_w", "Impp_a", "Vmpp_v"])

    if soiling:
        soiling_ratio = hsu_soiling(weather_df, pv_params)
        data["soiling_loss_perc"] = (1 - soiling_ratio) * 100

    if method == "singlediode" or method == "singlediode_regress":
        alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s = pv_params["alpha_sc"], pv_params["a_ref"], \
                                                           pv_params["I_L_ref"], pv_params["I_o_ref"], \
                                                           pv_params["R_sh_ref"], pv_params["R_s"]

        if method == "singlediode_regress":
            iv_curve_regressor = vi_singlediode_regressor(alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s,
                                                          store_pkl=True)

        for idx, row in tqdm(weather_df.iterrows(), total=weather_df.shape[0]):
            if row["Gi_w.m2"] != 0:
                # Get vi_curve at the system level
                if method == "singlediode":
                    vi_curve = vi_curve_singlediode(alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s,
                                                    effective_irradiance=row["Ee_w.m2"], temp_cell=row["Tc_C"])
                elif method == "singlediode_regress":
                    vi_curve = iv_curve_regressor(row["Ee_w.m2"], row["Tc_C"])

                if soiling:
                    vi_curve["i"] = vi_curve["i"] * soiling_ratio.loc[idx]
                vi_string = combine_series(dfs=[vi_curve] * pv_params["modules_serie"])
                vi_system = combine_parallel([vi_string] * pv_params["strings"])

                # curve_plot(vi_curve, legend="Single Module")
                # curve_plot(vi_system, legend="System")

                # Collect operating electrical points
                Pmpp, I_mpp, V_mpp = get_Pmpp(vi_system, VI_max=True)
                data.loc[idx, "Pmpp_w"] = Pmpp
                data.loc[idx, "Impp_a"] = I_mpp
                data.loc[idx, "Vmpp_v"] = V_mpp

    return data


def iv_regressor_full(regressor_i, regressor_v, effective_irradiance, temp_cell, ):
    vi_curve = pd.DataFrame(columns=["v", "i"])
    vi_curve["i"] = regressor_i.predict(np.array([[effective_irradiance, temp_cell]]))[0]
    vi_curve["v"] = regressor_v.predict(np.array([[effective_irradiance, temp_cell]]))[0]

    # Avoid issues when combining iv_curves
    if (vi_curve["i"] == 0).all():
        vi_curve["i"] = range(1000, 0, -1)
        vi_curve["i"] *= 1e-12

    return vi_curve.astype(float)


def vi_singlediode_regressor(alpha_sc: float, a_ref: float, I_L_ref: float, I_o_ref: float, R_sh_ref: float,
                             R_s: float, n_points: float = 1000, EgRef: float = 1.121,
                             dEgdT: float = -0.0002677, store_pkl=True) -> pd.DataFrame:
    """
    Get the IV curve KNN-regressor with effective irradiance and cell temperature as inputs.

     :param alpha_sc: The short-circuit current temperature coefficient of the  module in units of A/C.
     :param a_ref: The product of the usual diode ideality factor (n, unitless),
         number of cells in series (Ns), and cell thermal voltage at reference
         conditions, in units of V.
     :param IL: The light-generated current (or photocurrent) at reference conditions, in amperes.
     :param I_o_ref: The dark or diode reverse saturation current at reference conditions, in amperes.
     :param Rs: The series resistance at reference conditions, in ohms.
     :param Rsh: The shunt resistance at reference conditions, in ohms.
     :param n_points: Number of points in the desired IV curve
     :param EgRef: The energy bandgap at reference temperature in units of eV.
         1.121 eV for crystalline silicon. EgRef must be >0.  For parameters
         from the SAM CEC module database, EgRef=1.121 is implicit for all
         cell types in the parameter estimation algorithm used by NREL.
     :param dEgdT:  The temperature dependence of the energy bandgap at reference
         conditions in units of 1/K. May be either a scalar value
         (e.g. -0.0002677 as in [1]_) or a DataFrame (this may be useful if
         dEgdT is a modeled as a function of temperature). For parameters from
         the SAM CEC module database, dEgdT=-0.0002677 is implicit for all cell
         types in the parameter estimation algorithm used by NREL.
    :param store: Store/use the pickle file

     :return: f_Pmpp, f_Impp, f_Vmpp interpolation functions
     """

    path_store_i = str(ROOT / "data" / "regressor_i.pkl")
    path_store_v = str(ROOT / "data" / "regressor_v.pkl")
    if store_pkl and os.path.exists(path_store_i) and os.path.exists(path_store_v):
        with open(path_store_i, "rb") as input_file:
            regressor_i = pickle.load(input_file)
        with open(path_store_v, "rb") as input_file:
            regressor_v = pickle.load(input_file)

    else:
        index = pd.MultiIndex.from_product([list(range(0, 2000, 5)), range(-20, 100, 1)],
                                           names=["Ee", "Tc"])
        i_data = pd.DataFrame(index=index, columns=range(n_points))
        v_data = pd.DataFrame(index=index, columns=range(n_points))
        for effective_irradiance, temp_cell in tqdm(index, desc="Mpp interpolation"):
            if effective_irradiance != 0:
                vi_curve = vi_curve_singlediode(alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, n_points,
                                                effective_irradiance,
                                                temp_cell, EgRef, dEgdT)
                i_data.loc[(effective_irradiance, temp_cell), :] = vi_curve["i"]
                v_data.loc[(effective_irradiance, temp_cell), :] = vi_curve["v"]

            else:
                i_data.loc[(effective_irradiance, temp_cell), :] = 0
                v_data.loc[(effective_irradiance, temp_cell), :] = np.nan

        X_i = i_data.reset_index()[["Ee", "Tc"]]
        v_data = v_data.dropna()
        X_v = v_data.reset_index()[["Ee", "Tc"]]
        regressor_i = KNeighborsRegressor(n_neighbors=1).fit(X_i.values, i_data)
        regressor_v = KNeighborsRegressor(n_neighbors=1).fit(X_v.values, v_data)

        if store_pkl and os.path.exists(str(ROOT / "data")):
            with open(path_store_i, 'wb') as handle:
                pickle.dump(regressor_i, handle)
            with open(path_store_v, 'wb') as handle:
                pickle.dump(regressor_v, handle)

    iv_regressor = partial(iv_regressor_full, regressor_i, regressor_v)
    return iv_regressor


def vi_curve_singlediode(alpha_sc: float, a_ref: float, I_L_ref: float, I_o_ref: float, R_sh_ref: float, R_s: float,
                         n_points: float = 1000, effective_irradiance: float = 1000, temp_cell: float = 25,
                         EgRef: float = 1.121,
                         dEgdT: float = -0.0002677) -> pd.DataFrame:
    """
    Draw the IV curve according to DeSoto method and the single Diode model.

     :param alpha_sc: The short-circuit current temperature coefficient of the  module in units of A/C.
     :param a_ref: The product of the usual diode ideality factor (n, unitless),
         number of cells in series (Ns), and cell thermal voltage at reference
         conditions, in units of V.
     :param IL: The light-generated current (or photocurrent) at reference conditions, in amperes.
     :param I_o_ref: The dark or diode reverse saturation current at reference conditions, in amperes.
     :param Rs: The series resistance at reference conditions, in ohms.
     :param Rsh: The shunt resistance at reference conditions, in ohms.
     :param n_points: Number of points in the desired IV curve
     :param effective_irradiance: The irradiance (W/m2) that is converted to photocurrent.
     :param temp_cell:  The average cell temperature of cells within a module in C.
     :param EgRef: The energy bandgap at reference temperature in units of eV.
         1.121 eV for crystalline silicon. EgRef must be >0.  For parameters
         from the SAM CEC module database, EgRef=1.121 is implicit for all
         cell types in the parameter estimation algorithm used by NREL.
     :param dEgdT:  The temperature dependence of the energy bandgap at reference
         conditions in units of 1/K. May be either a scalar value
         (e.g. -0.0002677 as in [1]_) or a DataFrame (this may be useful if
         dEgdT is a modeled as a function of temperature). For parameters from
         the SAM CEC module database, dEgdT=-0.0002677 is implicit for all cell
         types in the parameter estimation algorithm used by NREL.

     :return: IV curve
         * i - IV curve current in amperes.
         * v - IV curve voltage in volts.
     """
    # adjust the reference parameters according to the operating conditions (effective_irradiance, temp_cell)
    # using the De Soto model:
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(effective_irradiance, temp_cell,
                                                         alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s,
                                                         EgRef, dEgdT)

    # Solve the single-diode equation to obtain a photovoltaic IV curve.
    curve_info = pvsystem.singlediode(
        photocurrent=IL,
        saturation_current=I0,
        resistance_series=Rs,
        resistance_shunt=Rsh,
        nNsVth=nNsVth,
        ivcurve_pnts=n_points,
        method='lambertw'
    )

    curve_df = pd.DataFrame(curve_info)[["v", "i"]]

    return curve_df


def interpolate_i(df, i):
    """convenience wrapper around scipy.interpolate.interp1d"""
    f_interp = interp1d(df['i'], df['v'], kind='linear', fill_value='extrapolate')
    return f_interp(i)


def interpolate_v(df, v):
    """convenience wrapper around scipy.interpolate.interp1d"""
    f_interp = interp1d(df['v'], df['i'], kind='linear', fill_value='extrapolate')
    return f_interp(v)


def combine_parallel(dfs, n_points=1000):
    """
    Combine IV curves in parallel by aligning voltages and summing currents.
    The current range is based on the first curve's voltage range.
    """
    df1 = dfs[0]
    vmin = df1['v'].min()
    vmax = df1['v'].max()
    v = np.linspace(vmin, vmax, n_points)
    i = 0
    for df2 in dfs:
        i_cell = interpolate_v(df2, v)
        i += i_cell
    i[i < 0] = np.nan
    return pd.DataFrame({'i': i, 'v': v})


def combine_series(dfs, n_points=1000):
    """
    Combine IV curves in series by aligning currents and summing voltages.
    The current range is based on the first curve's current range.
    """
    df1 = dfs[0]
    imin = df1['i'].min()
    imax = df1['i'].max()
    i = np.linspace(imin, imax, n_points)
    v = 0

    for df2 in dfs:
        v_cell = interpolate_i(df2, i)
        v += v_cell
    v[v < 0] = np.nan

    return pd.DataFrame({'i': i, 'v': v})


def get_Pmpp(curve_tmp, VI_max=False):
    p = curve_tmp["v"] * curve_tmp["i"]
    Pmpp = p.max()

    if VI_max:
        I_mpp, V_mpp = curve_tmp.loc[p.idxmax(), ["i", "v"]].values
        return Pmpp, I_mpp, V_mpp
    else:
        return Pmpp


def curve_plot(curve, legend: str = None, show_Pmax=True):
    plt.plot(curve["v"], curve["i"], label=legend)
    if show_Pmax:
        Pmax, I_max, V_max = get_Pmpp(curve, VI_max=True)
        plt.plot(V_max, I_max, label=(legend + "-Pmpp" if legend is not None else "Pmpp"), color="red", marker="o")
    plt.title("VI Curve")
    plt.ylabel("Intensity [A]")
    plt.xlabel("Voltage [V]")
    plt.legend()


if __name__ == "__main__":
    import pvlib

    pv_params = pvlib.pvsystem.retrieve_sam('cecmod')['Canadian_Solar_Inc__CS5P_220M']
    alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s = pv_params["alpha_sc"], pv_params["a_ref"], \
                                                       pv_params["I_L_ref"], pv_params["I_o_ref"], \
                                                       pv_params["R_sh_ref"], pv_params["R_s"]
    vi_curve = vi_curve_singlediode(alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s)
    vi_curve2 = vi_curve_singlediode(alpha_sc, a_ref, I_L_ref * 1.02, I_o_ref, R_sh_ref, R_s)
    from src.utils.helio_fmt import setup_helio_plt

    setup_helio_plt()
    curve_plot(vi_curve, legend="Before cleaning")
    curve_plot(vi_curve2, legend="After cleaning")
