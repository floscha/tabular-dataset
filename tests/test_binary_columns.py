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


def test_column_names_are_correctly_set():
    df = get_test_df()

    tds = TabularDataset(df, binary_columns=['B'])

    assert tds.binary.column_names == ['B']


def test_encode():
    df = get_test_df()

    tds = TabularDataset(df, binary_columns=['B'])
    tds.binary.encode()

    assert repr(tds.x) == repr(np.array([-1, 1, -1, np.nan]).reshape(-1, 1))


def test_counts_without_nan_values():
    df = get_test_df()

    tds = TabularDataset(df.dropna(), binary_columns=['B'])
    tds.binary.counts()

    assert repr(tds.x[:, 1]) == repr(np.array([2., 1., 2.]))


def test_counts_without_nan_values():
    df = get_test_df()

    tds = TabularDataset(df, binary_columns=['B'])
    tds.binary.encode()
    tds.binary.impute()
    tds.binary.counts()

    assert repr(tds.x[:, 1]) == repr(np.array([2., 1., 2., 1.]))


def test_impute():
    df = get_test_df()

    tds = TabularDataset(df, binary_columns=['B'])
    tds.binary.impute()

    assert repr(tds.x) == repr(np.array([0, 1, 0, 0.]).reshape(-1, 1))


if __name__ == '__main__':
    unittest.main()
