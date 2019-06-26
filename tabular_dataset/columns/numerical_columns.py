from typing import Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.transformations.numerical import impute, normalize


class NumericalColumns(AbstractColumns):
  def __init__(self, ds, column_names):
    super().__init__(ds, column_names)

    self._scaler = None

  def impute(self, columns: Optional[list] = None, method: str = 'median'):
    self.lineage.append((impute,
                         {'columns': columns or self.column_names,
                          'method': method}))
    return self.ds  # For fluent API

  def normalize(self, columns: Optional[list] = None, method: str = 'minmax'):
    self.lineage.append((normalize,
                         {'columns': columns or self.column_names,
                          'scaler': self._scaler,
                          'method': method}))
    return self.ds  # For fluent API
