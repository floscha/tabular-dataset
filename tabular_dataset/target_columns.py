from tabular_dataset.util import encode_categorical_columns


class TargetColumns:
  def __init__(self, ds, column_name):
    self.ds = ds
    self.column_names = [column_name]

    self._encoders = {}

  def encode(self, one_hot: bool = False):
    encode_categorical_columns(self, one_hot)

    return self.ds  # For fluent API
