import unittest

import numpy as np
import pandas as pd

from tabular_dataset import TabularDataset
from tabular_dataset.columns import (BinaryColumns, CategoricalColumns,
                                     NumericalColumns)


def get_test_df():
    return pd.DataFrame({
        'A': [1, 2, 3, np.nan],
        'B': [0, 1, 0, np.nan],
        'C': list('abba'),
        'target': list('xyzx')
    })


def test_repr():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], target_column='target')
    repr_output = repr(tds)

    assert repr_output == ("TabularDataset (4 rows)\n" +
                           "\tNumerical Columns: ['A']\n" +
                           "\tBinary Columns: ['B']\n" +
                           "\tCategorical Columns: ['C']\n" +
                           "\tTarget Column: 'target'")


def test_repr_with_multiple_target_columns():
    df = get_test_df()

    tds = TabularDataset(df, target_columns=['A', 'B'])
    repr_output = repr(tds)

    assert repr_output == ("TabularDataset (4 rows)\n" +
                           "\tTarget Columns: ['A', 'B']")


def test_x():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], target_column='target')

    assert repr(tds.x) == repr(df[['A', 'B', 'C']].values)


def test_y():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], target_column='target')

    assert repr(tds.y) == repr(df[['target']].values)


def test_x_train():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], target_column='target')

    assert repr(tds.x_train) == repr(df[['A', 'B', 'C']].values)


def test_y_train():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], target_column='target')

    assert repr(tds.y_train) == repr(df[['target']].values)


def test_x_test():
    df = get_test_df()
    test_data = df.iloc[-2:]

    tds = TabularDataset(df, test_data=test_data, numerical_columns=['A'],
                         binary_columns=['B'], categorical_columns=['C'],
                         target_column='target')

    assert repr(tds.x_test) == repr(test_data[['A', 'B', 'C']].values)


def test_y_test():
    df = get_test_df()
    test_data = df.iloc[-2:]

    tds = TabularDataset(df, test_data=test_data, numerical_columns=['A'],
                         binary_columns=['B'], categorical_columns=['C'],
                         target_column='target')

    assert repr(tds.y_test) == repr(test_data[['target']].values)


def test_num_abbreviation():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], target_column='target')

    assert isinstance(tds.num, NumericalColumns)


def test_bin_abbreviation():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], target_column='target')

    assert isinstance(tds.bin, BinaryColumns)


def test_cat_abbreviation():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], target_column='target')

    assert isinstance(tds.cat, CategoricalColumns)


if __name__ == '__main__':
    unittest.main()
