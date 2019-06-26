from typing import Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.columns.decorator import transformation
from tabular_dataset.transformations.categorical import (encode, hash, impute,
                                                         one_hot)


class CategoricalColumns(AbstractColumns):
  def __init__(self, ds, column_names):
    super().__init__(ds, column_names)

    self._encoders = {}

  @transformation
  def impute(self, columns: Optional[list] = None, method: str = 'unk'):
    return impute(method=method)

  @transformation
  def encode(self, columns: Optional[list] = None):
    return encode(encoders=self._encoders)

  @transformation
  def hash(self, columns: Optional[list] = None, bins: Optional[int] = None):
    return hash(bins=bins)

  @transformation
  def one_hot(self, columns: Optional[list] = None):
    return one_hot()
