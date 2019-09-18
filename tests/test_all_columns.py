import unittest

import numpy as np
import pandas as pd

from tabular_dataset import TabularDataset


def get_test_df():
    return pd.DataFrame({
        'A': [1, 2, 3, np.nan],
        'B': [0, 1, 0, np.nan],
        'C': list('abba'),
        'D': [0, 0, 1, 1],
        'target': list('xyzx')
    })


def test_column_names_are_correctly_set():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], datetime_columns=['D'])

    assert tds.all.column_names == ['A', 'B', 'C', 'D']


def test_len_property():
    df = get_test_df()
    tds = TabularDataset(df, numerical_columns=['A'], binary_columns=['B'],
                         categorical_columns=['C'], datetime_columns=['D'])

    length = len(tds.all)

    assert length == 4


if __name__ == '__main__':
    unittest.main()
