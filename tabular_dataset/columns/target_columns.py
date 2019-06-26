from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.transformations.categorical import encode, one_hot


class TargetColumns(AbstractColumns):
  def __init__(self, ds, column_names):
    super().__init__(ds, column_names)

    self._encoders = {}

  def encode(self):
    self.lineage.append((encode,
                         {'columns': self.column_names,
                          'encoders': self._encoders}))
    return self.ds  # For fluent API

  def one_hot(self):
    self.lineage.append((one_hot,
                         {'columns': self.column_names}))
    return self.ds  # For fluent API
