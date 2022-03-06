"""Using the code of songulerdem
https://www.kaggle.com/songulerdem/health-insurance-cross-sell-prediction-xgboost
"""

from typing import Any, Dict

import pandas as pd
import logging
import numpy as np

# categoricals: gender, region code, driving license, previously insured,
#   vehicle damage,
# ordinal: vehicle age

def trim_outliers(x: pd.Series, q1=0.05, q3=0.95) -> pd.Series:
    quartile1 = x.quantile(q1)
    quartile3 = x.quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 3 * interquantile_range
    low_limit = quartile1 - 3 * interquantile_range

    # cap (tukey's fences)
    return x.clip(lower=low_limit, upper=up_limit)


def ordinal_vehicle_age(x_veh_age: pd.Series) -> pd.Series: 
    x_veh_age.loc[x_veh_age == "< 1 Year"] = 1
    x_veh_age.loc[x_veh_age == "1-2 Year"] = 2
    x_veh_age.loc[x_veh_age == "> 2 Years"] = 3
    x_veh_age = x_veh_age.fillna(0)
    return x_veh_age


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data.

    Args:
        df: Raw data.
    Returns:
        Preprocessed data
    """

    df["Gender"] = df["Gender"] == "Female"
    df["Vehicle_Damage"] = df["Vehicle_Damage"] == "Yes"
    df["Vehicle_Age"] = ordinal_vehicle_age(df["Vehicle_Age"])

    # trim outliers
    df["Annual_Premium"] = trim_outliers(df["Annual_Premium"])

    # set variables as categoricals
    df["Region_Code"] = df["Region_Code"].astype(np.int32)
    df["Policy_Sales_Channel"] = df["Policy_Sales_Channel"].astype(np.int32)

    df = pd.get_dummies(df, columns=["Region_Code", "Policy_Sales_Channel"], 
                        prefix=["region", "channel"])
    
    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {df.columns.tolist()}")
    if np.any(df.isnull()):
        logger.info(df.isnull().sum().sort_values()[::-1])
        raise ValueError("Has nulls")
    else:
        return df
    