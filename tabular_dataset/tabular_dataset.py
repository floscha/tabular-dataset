from typing import List

import pandas as pd

from tabular_dataset.columns import (AllColumns,  BinaryColumns,
                                     CategoricalColumns,  NumericalColumns,
                                     TargetColumns)


class TabularDataset:
    def __init__(self, data, numerical_columns=None, binary_columns=None,
                 categorical_columns=None, target_column: str = None,
                 target_columns: List[str] = None):
        self.df = data  # TODO Copy data to avoid changing the original object?

        self.numerical = NumericalColumns(self, numerical_columns or [])
        self.binary = BinaryColumns(self, binary_columns or [])
        self.categorical = CategoricalColumns(self, categorical_columns or [])

        self.all = AllColumns(self)

        if target_column:
            if target_columns:
                raise ValueError("Only 'target_column' or 'target_columns' " +
                                 "can be set")
            else:
                self.target = TargetColumns(self, [target_column])
        elif target_columns:
            self.target = TargetColumns(self, target_columns)

    def __repr__(self):
        s = [f'TabularDataset ({self.df.shape[0]} rows)']
        if self.numerical:
            s.append(f'\tNumerical Columns: {self.numerical.column_names}')
        if self.binary:
            s.append(f'\tBinary Columns: {self.binary.column_names}')
        if self.categorical:
            s.append(f'\tCategorical Columns: {self.categorical.column_names}')
        if self.target:
            if len(self.target) == 1:
                s.append(f'\tTarget Column: {self.target.column_names[0]!r}')
            else:
                s.append(f'\tTarget Columns: {self.target.column_names}')

        return '\n'.join(s)

    @property
    def x(self):
        return pd.concat([self.numerical.transform(),
                          self.binary.transform(),
                          self.categorical.transform()],
                         axis=1).values

    @property
    def y(self):
        return self.target.transform().values
