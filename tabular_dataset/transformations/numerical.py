from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
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
def scale(df: pd.DataFrame, columns: List[str],
          scalers: List[MinMaxScaler], fit: bool) -> pd.DataFrame:
    scaler = scalers[0]
    if fit:
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        scalers[0] = scaler
    else:
        if scaler is None:
            raise ValueError("Scaler has to be fit first")
        df[columns] = scaler.transform(df[columns])
    return df


@transformation
def normalize(df: pd.DataFrame, columns: List[str],
              stats: Dict[str, Tuple[float, float]], fit: bool) \
              -> pd.DataFrame:
    for column_name in columns:
        if fit:
            column_stats = df[column_name].mean(), df[column_name].std()
            stats[column_name] = column_stats
        else:
            column_stats = stats[column_name]
        mean, std = column_stats
        df[column_name] = (df[column_name] - mean) / std
    return df


@transformation
def log(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply log-transformation to numerical columns.

    By defintion of the log operation, no negative values are supported.
    A 1 is added to all values to make the transform work for 0 values as well.
    """
    df[columns] = df[columns].apply(lambda x: np.log(1 + x))
    return df


@transformation
def power(df: pd.DataFrame, columns: List[str], exponent: int) -> pd.DataFrame:
    """Apply power-transformation to numerical columns."""
    df[columns] = df[columns].apply(lambda x: np.power(x, exponent))
    return df


@transformation
def add_ranks(df: pd.DataFrame, columns: List[str],
              method: Optional[str] = 'average') -> pd.DataFrame:
    """Assign ranks to data, dealing with ties appropriately.

    Args:
        method: The method used to assign ranks to tied elements. The options
            are ‘average’, ‘min’, ‘max’, ‘dense’ and ‘ordinal’:

            ‘average’: The average of the ranks that would have been assigned
            to all the tied values is assigned to each value.

            ‘min’: The minimum of the ranks that would have been assigned to
            all the tied values is assigned to each value. (This is also
            referred to as “competition” ranking.)

            ‘max’: The maximum of the ranks that would have been assigned to
            all the tied values is assigned to each value.

            ‘dense’: Like ‘min’, but the rank of the next highest element is
            assigned the rank immediately after those assigned to the tied
            elements.

            ‘ordinal’: All values are given a distinct rank, corresponding to
            the order that the values occur in a.

            The default is ‘average’.
    """
    for column_name in columns:
        df[column_name + '_rank'] = rankdata(df[column_name], method=method)
    return df


@transformation
def remove_outliers(df: pd.DataFrame, columns: List[str],
                    lower_percentile: int = 1, upper_percentile: int = 99) \
                    -> pd.DataFrame:
    """Removes outliers below and above given percentiles.

    Technically, instead of removing the values, they are replaced by an upper
    or lower bound.

    TODO: Right now, bounds for train and test sets are calculated separately.
          Instead, remember the bounds from training so that they can be
          re-used during test time, similar to `stats` for `normalize`.
    """
    for column_name in columns:
        lower_bound, upper_bound = np.percentile(df[column_name],
                                                 [lower_percentile,
                                                  upper_percentile])
        df[column_name] = np.clip(df[column_name], lower_bound, upper_bound)
    return df
