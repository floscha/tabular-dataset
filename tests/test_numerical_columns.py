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

    tds = TabularDataset(df, numerical_columns=['A'])

    assert tds.numerical.column_names == ['A']


def test_normalize():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'])
    tds.numerical.normalize()

    assert repr(tds.x) == repr(np.array([[0.], [0.5], [1.], [np.nan]]))


def test_normalize_no_fit():
    df = get_test_df()
    test_data = df.iloc[-2:]

    tds = TabularDataset(df, test_data=test_data, numerical_columns=['A'])
    tds.numerical.normalize()

    _ = tds.x_train
    assert repr(tds.x_test) == repr(np.array([[1.], [np.nan]]))


def test_log():
    df = pd.DataFrame({'A': [-2, -1, 0, 1, 2, np.nan]})
    tds = TabularDataset(df, numerical_columns=['A'])
    expected_result = np.array([np.nan, -float('inf'), 0.000000, 0.693147,
                                1.098612, np.nan])

    # Ignore "divide by zero" warning for testing.
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        tds.numerical.log()
        actual_result = tds.x[:, 0]

    assert np.allclose(actual_result, expected_result, equal_nan=True)


def test_power():
    df = pd.DataFrame({'A': [-2, -1, 0, 1, 2, np.nan]})
    tds = TabularDataset(df, numerical_columns=['A'])
    expected_result = np.array([4., 1., 0., 1., 4., np.nan])

    tds.numerical.power(exponent=2)
    actual_result = tds.x[:, 0]

    assert np.allclose(actual_result, expected_result, equal_nan=True)


def test_ranks_with_default_method():
    df = pd.DataFrame({'A': [0, 2, 3, 2]})
    tds = TabularDataset(df, numerical_columns=['A'])
    expected_result = np.array([1., 2.5, 4., 2.5])

    tds.numerical.ranks()
    actual_result = tds.x[:, 1]

    assert np.allclose(actual_result, expected_result)


def test_impute_with_median():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'])
    tds.numerical.impute()


def test_impute_with_median_no_fit():
    df = get_test_df()
    test_data = df.iloc[-2:]

    tds = TabularDataset(df, test_data=test_data, numerical_columns=['A'])
    tds.numerical.impute()

    _ = tds.x_train
    assert repr(tds.x_test) == repr(np.array([[3], [2.]]))


def test_impute_with_mean():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'])
    tds.numerical.impute(method='mean')

    assert repr(tds.x) == repr(np.array([[1], [2], [3], [2.]]))


def test_impute_with_zero():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'])
    tds.numerical.impute(method='zero')

    assert repr(tds.x) == repr(np.array([[1], [2], [3], [0.]]))


def test_impute_column():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'])
    tds.numerical.impute(add_columns=True)

    assert list(tds.x[:, 1]) == [False, False, False, True]


def test_fluent_api():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'])
    (tds
     .numerical.impute()
     .numerical.normalize())

    assert repr(tds.x) == repr(np.array([[0.], [0.5], [1.], [0.5]]))


if __name__ == '__main__':
    unittest.main()
