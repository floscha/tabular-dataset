import unittest
from typing import Iterator

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


def test_train_test_split():
    df = get_test_df()

    tds = TabularDataset(df, categorical_columns=['A'], target_column='target')
    tds.categorical.impute()
    tds.categorical.encode(add_unk_category=True)
    tds.categorical.one_hot()
    x_train, x_test, y_train, y_test = tds.train_test_split(test_size=0.25,
                                                            shuffle=False)

    assert x_train.shape == (3, 4)
    assert x_test.shape == (1, 4)
    assert y_train.shape == (3, 1)
    assert y_test.shape == (1, 1)


def test_k_fold_cross_validation():
    df = get_test_df()
    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], target_column='target')

    cv_iterator = tds.split(n_splits=4)

    assert isinstance(cv_iterator, Iterator)

    cv_list = list(cv_iterator)
    assert len(cv_list) == 4

    for fold in cv_list:
        x_train, x_test, y_train, y_test = fold
        assert len(x_train) == 3
        assert len(x_test) == 1
        assert len(y_train) == 3
        assert len(y_test) == 1


if __name__ == '__main__':
    unittest.main()
