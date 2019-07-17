from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.columns.decorator import transformation
from tabular_dataset.transformations.categorical import encode, one_hot


class TargetColumns(AbstractColumns):
    def __init__(self, ds, column_names):
        super().__init__(ds, column_names)

        self._categorical_encoders = {}
        self._one_hot_encoders = {}

    @transformation
    def encode(self):
        return encode(encoders=self._categorical_encoders)

    @transformation
    def one_hot(self):
        return one_hot(encoders=self._one_hot_encoders)
