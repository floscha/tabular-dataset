from typing import Optional

from tabular_dataset.util import (encode_categorical_columns,
                                  hash_categorical_columns,
                                  one_hot_categorical_columns)


UNK_TOKEN = '<UNK>'


class CategoricalColumns:
  def __init__(self, ds, column_names):
    self.ds = ds
    self.column_names = column_names

    self._encoders = {}

  def impute(self, method='unk'):
    if method == 'unk':
      fill_values = UNK_TOKEN
    elif method == 'mode':
      fill_values = self.ds.df[self.column_names].mode().iloc[0, :]
    else:
      raise ValueError("Method not supported")

    self.ds.df[self.column_names] = (self.ds.df[self.column_names]
                                     .fillna(fill_values))

    return self.ds  # For fluent API

  def encode(self, columns: Optional[list] = None):
    encode_categorical_columns(self, columns or self.column_names)

    return self.ds  # For fluent API

  def hash(self, columns: Optional[list] = None, bins: Optional[int] = None):
    if bins is None:
      raise ValueError("'bins' argument needs to be set")
    hash_categorical_columns(self, columns or self.column_names, bins)

    return self.ds  # For fluent API

  def one_hot(self, columns: Optional[list] = None):
    one_hot_categorical_columns(self, columns or self.column_names)

    return self.ds  # For fluent API
