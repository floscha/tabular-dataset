from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tabular_dataset.transformations.common import add_imputed_columns
from tabular_dataset.transformations.decorator import transformation


@transformation
def impute(df: pd.DataFrame, columns: List[str], impute_values: List[float],
           method: Optional[str] = None, fit: bool = True,
           add_columns: bool = False) -> pd.DataFrame:
    if fit:
        if not method:
            raise ValueError("'method' has to be specified when fitting")
        if impute_values:
            raise ValueError("'impute value' argument cannot be used when " +
                             "fitting")

        if method == 'median':
            impute_values.extend(df.median())
        elif method == 'mean':
            impute_values.extend(df.mean())
        elif method == 'zero':
            impute_values.append(0)
        else:
            raise ValueError("Method not supported")
    else:
        if method:
            pass  # TODO: Print warning instead
            # raise ValueError("'method' argument cannot be used when fitting")
        if not impute_values:
            raise ValueError("'impute value' has to be specified when fitting")

    if add_columns:
        add_imputed_columns(df, columns)

    df[columns] = df[columns].fillna(impute_values[0]
                                     if len(impute_values) == 1
                                     else impute_values)
    return df


@transformation
def normalize(df: pd.DataFrame, columns: List[str],
              scalers: List[MinMaxScaler], method: str, fit: bool) \
              -> pd.DataFrame:
    scaler = scalers[0]
    if fit:
        if method == 'minmax':
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
            scalers[0] = scaler
        else:
            raise ValueError("Method not supported")
    else:
        if scaler is None:
            raise ValueError("Scaler has to be fit first")
        df[columns] = scaler.transform(df[columns])

    return df


@transformation
def log(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply log-transformation to numerical columns.

    By defintion of the log operation, no negative values are supported.
    A 1 is added to all values to make the transform work for 0 values as well.
    """
    return df.apply(lambda x: np.log(1 + x))
