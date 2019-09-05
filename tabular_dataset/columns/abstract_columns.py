import inspect
from typing import List, Optional

import pandas as pd


class AbstractColumns:
    try:
        from tabular_dataset import TabularDataset
    except: pass  # noqa: E722

    def __init__(self, ds: 'TabularDataset', column_names: List[str]):
        self.ds = ds
        self.column_names = column_names

        self.lineage = []  # type: list

    def __bool__(self):
        return len(self) > 0

    def __len__(self):
        return len(self.column_names)

    def transform(self, data: Optional[pd.DataFrame] = None, test: bool = False):
        if data is not None:
            df = data
        else:
            if test:
                if self.ds.test_df is None:
                    raise ValueError("'test_data' arguments needs to be set " +
                                     "for TabularDataset")
                df = self.ds.test_df
            else:
                df = self.ds.df
        df = df[self.column_names].copy()

        for transformation_fn in self.lineage:
            parameters = inspect.signature(transformation_fn).parameters
            df = (transformation_fn(df, fit=not test) if 'fit' in parameters
                  else transformation_fn(df))

        return df
