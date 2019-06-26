from typing import Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.columns.decorator import transformation
from tabular_dataset.transformations.numerical import impute, normalize


class NumericalColumns(AbstractColumns):
    def __init__(self, ds, column_names):
        super().__init__(ds, column_names)

        self._scaler = None

    @transformation
    def impute(self, columns: Optional[list] = None, method: str = 'median'):
        return impute(method=method)

    @transformation
    def normalize(self, columns: Optional[list] = None,
                  method: str = 'minmax'):
        return normalize(scaler=self._scaler, method=method)
