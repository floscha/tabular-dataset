from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from tabular_dataset.columns import (AllColumns,  BinaryColumns,
                                     CategoricalColumns,  NumericalColumns,
                                     TargetColumns)


class TabularDataset:
    def __init__(self, data, test_data: Optional[pd.DataFrame] = pd.DataFrame,
                 numerical_columns: Optional[pd.DataFrame] = None,
                 binary_columns: Optional[pd.DataFrame] = None,
                 categorical_columns: Optional[pd.DataFrame] = None,
                 target_column: Optional[str] = None,
                 target_columns: Optional[List[str]] = None):
        self.df = data  # TODO Copy data to avoid changing the original object?
        self.test_df = test_data

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

    def __repr__(self) -> str:
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

    # --- Various getters for data to be used by machine learning algorithms

    @property
    def x(self) -> np.array:
        return pd.concat([self.numerical.transform(),
                          self.binary.transform(),
                          self.categorical.transform()],
                         axis=1).values

    @property
    def y(self) -> np.array:
        return self.target.transform().values

    @property
    def x_train(self) -> np.array:
        return self.x

    @property
    def y_train(self) -> np.array:
        return self.y

    @property
    def x_test(self) -> np.array:
        return pd.concat([self.numerical.transform(test=True),
                          self.binary.transform(test=True),
                          self.categorical.transform(test=True)],
                         axis=1).values

    @property
    def y_test(self) -> np.array:
        return self.target.transform(test=True).values

    def split(self, n_splits: int = 2, random_state: Optional[int] = None,
              shuffle: bool = False):
        """Create a number of splits for K-fold cross validation.

        Uses an iterator to not create all folds at once. Otherwise, e.g. 1GB
        of data would turn into 10GB with n_splits = 10.

        Args:
            n_splits: The number of splits to divide the dataset into.
            random_state: A seed to reproduce splits.
            shuffle: Shuffle the data when True or not when False.
        """
        # Store x and y to not compute them again with each fold.
        x, y = self.x, self.y
        kf = KFold(n_splits=n_splits, random_state=random_state,
                   shuffle=shuffle)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            yield x_train, x_test, y_train, y_test

    # --- Abbreviations for columns

    @property
    def num(self) -> NumericalColumns:
        return self.numerical

    @property
    def bin(self) -> BinaryColumns:
        return self.binary

    @property
    def cat(self) -> CategoricalColumns:
        return self.categorical
