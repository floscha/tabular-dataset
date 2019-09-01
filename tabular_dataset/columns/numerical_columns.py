from typing import List, Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.columns.decorator import transformation
from tabular_dataset.transformations.numerical import (add_ranks, impute, log,
                                                       normalize, power)


class NumericalColumns(AbstractColumns):
    try:
        from tabular_dataset import TabularDataset
    except: pass  # noqa: E722

    def __init__(self, ds: 'TabularDataset', column_names: List[str]):
        super().__init__(ds, column_names)

        self._impute_values = []  # type: list
        self._scalers = [None]  # type: list

    @transformation
    def impute(self, columns: Optional[List[str]] = None,
               method: str = 'median', add_columns: bool = False):
        return impute(method=method, impute_values=self._impute_values,
                      add_columns=add_columns)

    @transformation
    def normalize(self, columns: Optional[List[str]] = None,
                  method: str = 'minmax'):
        return normalize(scalers=self._scalers, method=method)

    @transformation
    def log(self, columns: Optional[List[str]] = None):
        return log()

    @transformation
    def power(self, exponent: int, columns: Optional[List[str]] = None):
        return power(exponent=exponent)

    @transformation
    def ranks(self, columns: Optional[List[str]] = None,
              method: Optional[str] = 'average'):
        return add_ranks(method=method)
