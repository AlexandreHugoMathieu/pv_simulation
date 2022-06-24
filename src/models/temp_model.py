# Created by A. MATHIEU at 08/05/2022


def temp_NOTC(weather_df, pv_params):
    tnotc = pv_params["T_NOCT"]
    Tc = weather_df["Ta_C"] + weather_df["Ee_w.m2"] / 800 * (tnotc - 20)
    return Tc
