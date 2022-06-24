# Created by A. MATHIEU at 24/06/2022
import pandas as pd

from typing import Callable


def simulation_ui(start_date: pd.Timestamp,
                  end_date: pd.Timestamp,
                  temp_model:  Callable,
                  elec_model:  Callable,
                  failure_params: dict = {}):
    return None
