from typing import Optional

import pandas as pd

from tabular_dataset.columns.abstract_columns import AbstractColumns


class AllColumns(AbstractColumns):
    try:
        from tabular_dataset import TabularDataset
    except: pass  # noqa: E722

    def __init__(self, ds: 'TabularDataset'):
        self.ds = ds

    def transform(self, data: Optional[pd.DataFrame] = None,
                  test: bool = False):
        return pd.concat([self.ds.numerical.transform(data=data, test=test),
                          self.ds.binary.transform(data=data, test=test),
                          self.ds.categorical.transform(data=data, test=test),
                          self.ds.datetime.transform(data=data, test=test)],
                         axis=1)

    @property
    def column_names(self):
        return (self.ds.numerical.column_names +
                self.ds.binary.column_names +
                self.ds.categorical.column_names +
                self.ds.datetime.column_names)
