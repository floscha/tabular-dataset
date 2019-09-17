from typing import List, Optional

import pandas as pd

from tabular_dataset.transformations.decorator import transformation


@transformation
def encode(df: pd.DataFrame, columns: List[str],
           datetime_components: Optional[List[str]] = None) -> pd.DataFrame:
    default_date_components = ['year', 'month', 'weekofyear', 'day',
                               'dayofweek']
    default_time_components = ['hour', 'minute', 'second']
    datetime_components = datetime_components or (default_date_components +
                                                  default_time_components)

    for column_name in columns:
        df[column_name] = pd.to_datetime(df[column_name])
        for component in datetime_components:
            df[column_name + '_' + component] = df[
                column_name].dt.__getattribute__(component)

    return df.drop(columns=columns)
