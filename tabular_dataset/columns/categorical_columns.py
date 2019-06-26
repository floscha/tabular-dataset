from typing import Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.transformations.categorical import encode, hash, one_hot


UNK_TOKEN = '<UNK>'


class CategoricalColumns(AbstractColumns):
  def __init__(self, ds, column_names):
    super().__init__(ds, column_names)

    self._encoders = {}

  def _impute(self, df, columns: list, method: str):
    if method == 'unk':
      fill_values = UNK_TOKEN
    elif method == 'mode':
      fill_values = df.mode().iloc[0, :]
    else:
      raise ValueError("Method not supported")

    return df.fillna(fill_values)

  def impute(self, columns: Optional[list] = None, method: str = 'unk'):
    self.lineage.append((self._impute,
                         {'columns': columns or self.column_names,
                          'method': method}))
    return self.ds  # For fluent API

  def encode(self, columns: Optional[list] = None):
    self.lineage.append((encode,
                         {'columns': columns or self.column_names,
                          'encoders': self._encoders}))
    return self.ds  # For fluent API

  def hash(self, columns: Optional[list] = None, bins: Optional[int] = None):
    if bins is None:
      raise ValueError("'bins' argument needs to be set")
    self.lineage.append((hash,
                         {'columns': columns or self.column_names,
                          'bins': bins}))
    return self.ds  # For fluent API

  def one_hot(self, columns: Optional[list] = None):
    self.lineage.append((one_hot,
                         {'columns': columns or self.column_names}))
    return self.ds  # For fluent API
