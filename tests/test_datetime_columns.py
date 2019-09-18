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


def test_column_names_are_correctly_set():
    df = get_test_df()

    tds = TabularDataset(df, datetime_columns=['dt1', 'dt2'])

    assert tds.datetime.column_names == ['dt1', 'dt2']


def test_encode_defaults_are_correct2():
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
