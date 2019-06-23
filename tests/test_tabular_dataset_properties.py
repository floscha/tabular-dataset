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


def test_x():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'])

    assert tds.numerical.column_names == ['A']


def test_y():
    df = get_test_df()

    tds = TabularDataset(df, numerical_columns=['A'])

    assert tds.numerical.column_names == ['A']


if __name__ == '__main__':
    unittest.main()
