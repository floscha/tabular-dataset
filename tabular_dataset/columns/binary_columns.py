from typing import List, Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.columns.decorator import transformation
from tabular_dataset.transformations.binary import encode, impute
from tabular_dataset.transformations.categorical import counts, frequencies


class BinaryColumns(AbstractColumns):
    try:
        from tabular_dataset import TabularDataset
    except: pass  # noqa: E722

    def __init__(self, ds: 'TabularDataset', column_names: List[str]):
        super().__init__(ds, column_names)

    @transformation
    def encode(self, columns: Optional[List[str]] = None):
        return encode()

    @transformation
    def impute(self, columns: Optional[List[str]] = None):
        return impute()

    @transformation
    def counts(self, columns: Optional[List[str]] = None):
        return counts()

    @transformation
    def frequencies(self, columns: Optional[List[str]] = None):
        return frequencies()
