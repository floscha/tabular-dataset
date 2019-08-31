import unittest

import numpy as np
import pandas as pd

from tabular_dataset import TabularDataset


def get_test_df():
    return pd.DataFrame({
        'A': list('abba') + [np.nan, np.nan],
        'B': list('ccdd') + [np.nan, np.nan]
    })


def test_column_names_are_correctly_set():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])

    assert tds.categorical.column_names == ['A', 'B']


def test_encode():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.encode()

    assert repr(tds.x) == repr(np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                                         [2, 2], [2, 2]]))


def test_encode_with_selected_columns():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.encode(columns=['B'])

    assert repr(tds.x) == repr(np.array([['a', 0], ['b', 0], ['b', 1],
                                         ['a', 1], [np.nan, 2], [np.nan, 2]],
                                        dtype=np.object))


def test_encode_no_fit():
    df = get_test_df()
    test_data = df.iloc[-2:]

    tds = TabularDataset(df, test_data=test_data,
                         categorical_columns=['A', 'B'])
    tds.categorical.encode()

    _ = tds.x_train
    assert repr(tds.x_test) == repr(np.array([[2, 2], [2, 2]]))


def test_encode_with_hashing():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.encode()
    tds.categorical.hash(bins=2)

    assert repr(tds.x) == repr(np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                                         [0, 0], [0, 0]]))


def test_encode_one_hot():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.encode()
    tds.categorical.one_hot()

    assert repr(tds.x) == repr(np.array([[1., 0., 0., 1., 0., 0.],
                                         [0., 1., 0., 1., 0., 0.],
                                         [0., 1., 0., 0., 1., 0.],
                                         [1., 0., 0., 0., 1., 0.],
                                         [0., 0., 1., 0., 0., 1.],
                                         [0., 0., 1., 0., 0., 1.]]))


def test_encode_one_hot_drop_first():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.encode()
    tds.categorical.one_hot(drop_first=True)

    assert repr(tds.x) == repr(np.array([[0., 0., 0., 0.],
                                         [1., 0., 0., 0.],
                                         [1., 0., 1., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 1., 0., 1.],
                                         [0., 1., 0., 1.]]))


def test_encode_one_hot_no_fit():
    df = get_test_df()
    test_data = df.iloc[-2:]

    tds = TabularDataset(df, test_data=test_data,
                         categorical_columns=['A', 'B'])
    tds.categorical.encode()
    tds.categorical.one_hot()

    _ = tds.x_train
    assert repr(tds.x_test) == repr(np.array([[0., 0., 1., 0., 0., 1.],
                                              [0., 0., 1., 0., 0., 1.]]))


def test_encode_one_hot_with_hashing():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.encode()
    tds.categorical.hash(bins=2)
    tds.categorical.one_hot()

    assert repr(tds.x) == repr(np.array([[1., 0., 1., 0.],
                                         [0., 1., 1., 0.],
                                         [0., 1., 0., 1.],
                                         [1., 0., 0., 1.],
                                         [1., 0., 1., 0.],
                                         [1., 0., 1., 0.]]))


def test_counts_without_nan_values():
    df = get_test_df()

    tds = TabularDataset(df.dropna(), categorical_columns=['A', 'B'])
    tds.categorical.counts()

    assert list(tds.x[:, 2]) == [2, 2, 2, 2]
    assert list(tds.x[:, 3]) == [2, 2, 2, 2]


def test_counts_with_nan_values():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.encode()
    tds.categorical.impute()
    tds.categorical.counts()

    assert list(tds.x[:, 2]) == [2, 2, 2, 2, 2, 2]
    assert list(tds.x[:, 3]) == [2, 2, 2, 2, 2, 2]


def test_frequencies_without_nan_values():
    df = get_test_df()

    tds = TabularDataset(df.dropna(), categorical_columns=['A', 'B'])
    tds.categorical.frequencies()

    assert list(tds.x[:, 2]) == [0.5, 0.5, 0.5, 0.5]
    assert list(tds.x[:, 3]) == [0.5, 0.5, 0.5, 0.5]


def test_frequencies_with_nan_values():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.encode()
    tds.categorical.impute()
    tds.categorical.frequencies()

    assert list(tds.x[:, 2]) == [0.3333333333333333] * 6
    assert list(tds.x[:, 3]) == [0.3333333333333333] * 6


def test_impute_with_unk_token():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.impute()

    assert repr(tds.x) == repr(np.array([['a', 'c'], ['b', 'c'], ['b', 'd'],
                                         ['a', 'd'], ['<UNK>', '<UNK>'],
                                         ['<UNK>', '<UNK>']],
                                        dtype='object'))


def test_impute_with_mode():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.impute(method='mode')

    assert repr(tds.x) == repr(np.array([['a', 'c'], ['b', 'c'], ['b', 'd'],
                                         ['a', 'd'], ['a', 'c'], ['a', 'c']],
                                        dtype='object'))


def test_impute_with_mode_no_fit():
    df = get_test_df()
    test_data = df.iloc[-2:]

    tds = TabularDataset(df, test_data=test_data,
                         categorical_columns=['A', 'B'])
    tds.categorical.impute(method='mode')

    _ = tds.x_train
    assert repr(tds.x_test) == repr(np.array([['a', 'c'], ['a', 'c']],
                                             dtype='object'))


def test_impute_column():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A', 'B'])
    tds.categorical.impute(add_column=True)

    assert list(tds.x[:, 2]) == [False, False, False, False, True, True]
    assert list(tds.x[:, 3]) == [False, False, False, False, True, True]


if __name__ == '__main__':
    unittest.main()
