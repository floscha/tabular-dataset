from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from tabular_dataset.columns import (AllColumns,  BinaryColumns,
                                     CategoricalColumns,  NumericalColumns,
                                     TargetColumns)


class TabularDataset:
    def __init__(self, data, test_data: Optional[pd.DataFrame] = None,
                 numerical_columns: Optional[List[str]] = None,
                 binary_columns: Optional[List[str]] = None,
                 categorical_columns: Optional[List[str]] = None,
                 target_column: Optional[str] = None,
                 target_columns: Optional[List[str]] = None,
                 infer_column_types: bool = False):
        self.df = data  # TODO Copy data to avoid changing the original object?
        self.test_df = test_data

        numerical_columns = numerical_columns or []
        binary_columns = binary_columns or []
        categorical_columns = categorical_columns or []

        if target_column:
            if target_columns:
                raise ValueError("Only 'target_column' or 'target_columns' " +
                                 "can be set")
            else:
                target_columns = [target_column]
        else:
            if not target_columns:
                target_columns = []

        specified_columns = set(numerical_columns + binary_columns +
                                categorical_columns + target_columns)

        if infer_column_types:
            non_specified_columns = data.drop(columns=specified_columns)
            column_types = non_specified_columns.dtypes
            num_unique = non_specified_columns.nunique()

            foo = pd.concat([column_types, num_unique], axis=1)
            for column_name, (data_type, unique_values) in foo.iterrows():
                if data_type.name == 'bool':
                    binary_columns.append(column_name)
                elif data_type.name in ('int64', 'float64'):
                    if unique_values == 2:
                        binary_columns.append(column_name)
                    else:
                        numerical_columns.append(column_name)
                else:
                    # TODO Also handle dates later
                    categorical_columns.append(column_name)

        self.numerical = NumericalColumns(self, numerical_columns)
        self.binary = BinaryColumns(self, binary_columns)
        self.categorical = CategoricalColumns(self, categorical_columns)

        self.all = AllColumns(self)

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
        test_df = self.test_df
        return pd.concat([self.numerical.transform(data=test_df, test=True),
                          self.binary.transform(data=test_df, test=True),
                          self.categorical.transform(data=test_df, test=True)],
                         axis=1).values

    @property
    def y_test(self) -> np.array:
        test_df = self.test_df
        return self.target.transform(data=test_df, test=True).values

    def train_test_split(self, test_size: float = 0.1, shuffle: bool = True):
        """Split the tabular dataset into random train and test subsets."""
        x_train, x_test, y_train, y_test = train_test_split(
            self.df[self.all.column_names],
            self.df[self.target.column_names],
            test_size=test_size,
            shuffle=shuffle
        )
        x_train = pd.concat([self.numerical.transform(data=x_train),
                             self.binary.transform(data=x_train),
                             self.categorical.transform(data=x_train)],
                            axis=1).values
        x_test = pd.concat([self.numerical.transform(data=x_test, test=True),
                            self.binary.transform(data=x_test, test=True),
                            self.categorical.transform(data=x_test,
                                                       test=True)],
                           axis=1).values
        y_train = self.target.transform(data=y_train).values
        y_test = self.target.transform(data=y_test, test=True).values
        return x_train, x_test, y_train, y_test

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
