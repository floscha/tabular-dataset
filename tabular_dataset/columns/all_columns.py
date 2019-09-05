from typing import Optional

import pandas as pd

from tabular_dataset.columns.abstract_columns import AbstractColumns


class AllColumns(AbstractColumns):
    def __init__(self, ds):
        self.ds = ds

    def transform(self, data: Optional[pd.DataFrame] = None,
                  test: bool = False):
        raise TypeError(f"'AllColumns' does not support 'transform()'")

    @property
    def column_names(self):
        return (self.ds.numerical.column_names +
                self.ds.binary.column_names +
                self.ds.categorical.column_names)
