from typing import List, Optional

from tabular_dataset.columns.abstract_columns import AbstractColumns
from tabular_dataset.columns.decorator import transformation
from tabular_dataset.transformations.datetime import encode


class DatetimeColumns(AbstractColumns):
    try:
        from tabular_dataset import TabularDataset
    except: pass  # noqa: E722

    def __init__(self, ds: 'TabularDataset', column_names: List[str]):
        super().__init__(ds, column_names)

    @transformation
    def encode(self, columns: Optional[List[str]] = None,
               datetime_components: Optional[List[str]] = None):
        return encode(datetime_components=datetime_components)
