from typing import Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.columns.decorator import transformation
from tabular_dataset.transformations.categorical import (encode, hash, impute,
                                                         one_hot)


class CategoricalColumns(AbstractColumns):
    def __init__(self, ds, column_names):
        super().__init__(ds, column_names)

        self._impute_values = []
        self._categorical_encoders = {}
        self._one_hot_encoders = {}

    @transformation
    def impute(self, columns: Optional[list] = None, method: str = 'unk'):
        return impute(method=method, impute_values=self._impute_values)

    @transformation
    def encode(self, columns: Optional[list] = None):
        return encode(encoders=self._categorical_encoders)

    @transformation
    def hash(self, columns: Optional[list] = None, bins: Optional[int] = None):
        return hash(bins=bins)

    @transformation
    def one_hot(self, columns: Optional[list] = None,
                drop_first: bool = False):
        return one_hot(encoders=self._one_hot_encoders, drop_first=drop_first)
