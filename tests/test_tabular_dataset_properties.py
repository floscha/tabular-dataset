import unittest

import numpy as np
import pandas as pd

from tabular_dataset import TabularDataset


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


if __name__ == '__main__':
    unittest.main()
