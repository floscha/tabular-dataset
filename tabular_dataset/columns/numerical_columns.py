from typing import Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.columns.decorator import transformation
from tabular_dataset.transformations.numerical import impute, normalize


class NumericalColumns(AbstractColumns):
    def __init__(self, ds, column_names):
        super().__init__(ds, column_names)

        self._impute_values = []
        self._scaler = None

    @transformation
    def impute(self, columns: Optional[list] = None, method: str = 'median'):
        return impute(method=method, impute_values=self._impute_values)

    @transformation
    def normalize(self, columns: Optional[list] = None,
                  method: str = 'minmax'):
        # FIXME Passing scaler as a list to enable call be reference ist kind
        # of a hack, so find a better solution.
        return normalize(scalers=[self._scaler], method=method)
