import datetime
import unittest

import numpy as np
import pandas as pd

from tabular_dataset import TabularDataset


def get_test_df():
    return pd.DataFrame({
        'dt1': ['1/1/2018', np.datetime64('2018-01-01'),
                datetime.datetime(2018, 1, 1)],
        'dt2': ['1/1/2018', np.datetime64('2018-01-01'),
                datetime.datetime(2018, 1, 1)]
    })


def get_df_with_missing_values():
    return pd.DataFrame({'dt': ['1/1/2018', '1/1/2018', '1/2/2018', None]})



def test_column_names_are_correctly_set():
    df = get_test_df()

    tds = TabularDataset(df, datetime_columns=['dt1', 'dt2'])

    assert tds.datetime.column_names == ['dt1', 'dt2']


def test_impute():
    df = get_df_with_missing_values()
    tds = TabularDataset(df, datetime_columns=['dt'])

    tds.datetime.impute()

    assert tds.x[-1] == ['1/1/2018']


def test_impute_value_is_correctly_encoded():
    # This test assumes that the `encode()` method is correctly implemented.
    df = get_df_with_missing_values()
    tds = TabularDataset(df, datetime_columns=['dt'])

    tds.datetime.impute()
    tds.datetime.encode()

    assert repr(tds.x[-1]) == repr(np.array([2018, 1, 1, 1, 0, 0, 0, 0]))


def test_impute_with_test_data():
    # Assert that the median from the test set is not leaked into the train set
    # but instead the train set median is used.
    df = get_df_with_missing_values()
    train_data = df.iloc[[2]]
    test_data = df.iloc[[0, 1, 3]]
    tds = TabularDataset(train_data, test_data=test_data,
                         datetime_columns=['dt'])

    tds.datetime.impute()

    _ = tds.x_train
    assert tds.x_test[-1][0] == '1/2/2018'


def test_encode_defaults_are_correct():
    df = get_test_df()

    tds = TabularDataset(df, datetime_columns=['dt1', 'dt2'])
    tds.datetime.encode()

    assert repr(tds.x) == repr(np.array([[2018, 1, 1, 1, 0, 0, 0, 0,
                                          2018, 1, 1, 1, 0, 0, 0, 0],
                                         [2018, 1, 1, 1, 0, 0, 0, 0,
                                          2018, 1, 1, 1, 0, 0, 0, 0],
                                         [2018, 1, 1, 1, 0, 0, 0, 0,
                                          2018, 1, 1, 1, 0, 0, 0, 0]]))


def test_encode_with_custom_datetime_components():
    df = get_test_df()

    tds = TabularDataset(df, datetime_columns=['dt1', 'dt2'])
    tds.datetime.encode(datetime_components=['year', 'hour'])

    assert repr(tds.x) == repr(np.array([[2018, 0, 2018, 0],
                                         [2018, 0, 2018, 0],
                                         [2018, 0, 2018, 0]]))


if __name__ == '__main__':
    unittest.main()
