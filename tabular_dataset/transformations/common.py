from typing import List, Optional

import pandas as pd
from scipy.stats import rankdata

from tabular_dataset.transformations.decorator import transformation


def add_imputed_columns(df: pd.DataFrame, columns: List[str]) -> None:
    for column_name in columns:
        df[column_name + '_was_imputed'] = df[column_name].isna()


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
