from typing import Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.transformations.binary import encode, impute


class BinaryColumns(AbstractColumns):
  def __init__(self, ds, column_names):
    super().__init__(ds, column_names)

  def encode(self, columns: Optional[list] = None):
    self.lineage.append((encode,
                         {'columns': columns or self.column_names}))
    return self.ds  # For fluent API

  def impute(self, columns: Optional[list] = None):
    self.lineage.append((impute,
                         {'columns': columns or self.column_names}))
    return self.ds  # For fluent API
