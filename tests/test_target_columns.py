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

    tds = TabularDataset(df, target_column='target')

    assert tds.target.column_names == ['target']


def test_encode():
    df = get_test_df()

    tds = TabularDataset(df, target_column='target')
    tds.target.encode()

    assert repr(tds.y) == repr(np.array([0, 1, 2, 0]).reshape(-1, 1))


def test_encode_no_fit():
    df = get_test_df()
    test_data = df.iloc[-2:]

    tds = TabularDataset(df, test_data=test_data, target_column='target')
    tds.target.encode()

    _ = tds.y_train
    assert repr(tds.y_test) == repr(np.array([2, 0]).reshape(-1, 1))


def test_encode_one_hot():
    df = get_test_df()

    tds = TabularDataset(df, target_column='target')
    tds.target.encode()
    tds.target.one_hot()

    assert repr(tds.y) == repr(np.array([[1., 0., 0.],
                                         [0., 1., 0.],
                                         [0., 0., 1.],
                                         [1., 0., 0.]]))


def test_encode_one_hot_no_fit():
    df = get_test_df()
    test_data = df.iloc[-2:]

    tds = TabularDataset(df, test_data=test_data, target_column='target')
    tds.target.encode()
    tds.target.one_hot()

    _ = tds.y_train
    assert repr(tds.y_test) == repr(np.array([[0., 0., 1.],
                                              [1., 0., 0.]]))


if __name__ == '__main__':
    unittest.main()
