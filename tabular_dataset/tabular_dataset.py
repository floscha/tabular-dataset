import pandas as pd

from tabular_dataset.columns import (AllColumns,  BinaryColumns,
                                     CategoricalColumns,  NumericalColumns,
                                     TargetColumns)


class TabularDataset:
    def __init__(self, data, numerical_columns=None, binary_columns=None,
                 categorical_columns=None, target_column: str = None):
        self.df = data  # TODO Copy data to avoid changing the original object?

        self.numerical = NumericalColumns(self, numerical_columns or [])
        self.binary = BinaryColumns(self, binary_columns or [])
        self.categorical = CategoricalColumns(self, categorical_columns or [])

        self.all = AllColumns(self)

        self.target = TargetColumns(self, [target_column])

    @property
    def x(self):
        return pd.concat([self.numerical.transform(),
                          self.binary.transform(),
                          self.categorical.transform()],
                         axis=1).values

    @property
    def y(self):
        return self.target.transform().values
