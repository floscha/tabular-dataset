from typing import List

import pandas as pd


def add_imputed_columns(df: pd.DataFrame, columns: List[str]) -> None:
    for column_name in columns:
        df[column_name + '_was_imputed'] = df[column_name].isna()
