from tabular_dataset.util import (encode_categorical_columns,
                                  one_hot_categorical_columns)


class TargetColumns:
  def __init__(self, ds, column_name):
    self.ds = ds
    self.column_names = [column_name]

    self._encoders = {}

  def encode(self):
    encode_categorical_columns(self, self.column_names)

    return self.ds  # For fluent API

  def one_hot(self):
    one_hot_categorical_columns(self, self.column_names)

    return self.ds  # For fluent API
