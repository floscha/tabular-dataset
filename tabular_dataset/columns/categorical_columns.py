from typing import List, Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.columns.decorator import transformation
from tabular_dataset.transformations.categorical import (counts, encode,
                                                         frequencies, hash,
                                                         impute, one_hot)


class CategoricalColumns(AbstractColumns):
    try:
        from tabular_dataset import TabularDataset
    except: pass  # noqa: E722

    def __init__(self, ds: 'TabularDataset', column_names: List[str]):
        super().__init__(ds, column_names)

        self._impute_values = []  # type: list
        self._categorical_encoders = {}  # type: dict
        self._one_hot_encoders = {}  # type: dict

    @transformation
    def impute(self, columns: Optional[List[str]] = None, method: str = 'unk',
               add_columns: bool = False):
        return impute(method=method, impute_values=self._impute_values,
                      add_columns=add_columns)

    @transformation
    def encode(self, columns: Optional[List[str]] = None,
               add_unk_category: bool = False):
        return encode(encoders=self._categorical_encoders,
                      add_unk_category=add_unk_category)

    @transformation
    def hash(self, columns: Optional[List[str]] = None,
             bins: Optional[int] = None):
        return hash(bins=bins)

    @transformation
    def one_hot(self, columns: Optional[List[str]] = None,
                drop_first: bool = False):
        return one_hot(encoders=self._one_hot_encoders, drop_first=drop_first)

    @transformation
    def counts(self, columns: Optional[List[str]] = None):
        return counts()

    @transformation
    def frequencies(self, columns: Optional[List[str]] = None):
        return frequencies()
