from typing import Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.columns.decorator import transformation
from tabular_dataset.transformations.binary import encode, impute


class BinaryColumns(AbstractColumns):
    def __init__(self, ds, column_names):
        super().__init__(ds, column_names)

    @transformation
    def encode(self, columns: Optional[list] = None):
        return encode()

    @transformation
    def impute(self, columns: Optional[list] = None):
        return impute()
