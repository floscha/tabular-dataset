from typing import Optional

from tabular_dataset.util import encode_categorical_columns


UNK_TOKEN = '<UNK>'


class CategoricalColumns:
  def __init__(self, ds, column_names):
    self.ds = ds
    self.column_names = column_names

    self._encoders = {}

  def encode(self, one_hot: bool = False, hash_bins: Optional[int] = None):
    encode_categorical_columns(self, one_hot, hash_bins)

    return self.ds  # For fluent API

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
