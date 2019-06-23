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

  tds = TabularDataset(df, categorical_columns=['C'])

  assert tds.categorical.column_names == ['C']


def test_encode():
  df = get_test_df()

  tds = TabularDataset(df, categorical_columns=['C'])
  tds.categorical.encode()

  assert repr(tds.x) == repr(np.array([0, 1, 1, 0]).reshape(-1, 1))


def test_encode_one_hot():
  df = get_test_df()

  tds = TabularDataset(df, categorical_columns=['C'])
  tds.categorical.encode(one_hot=True)

  assert repr(tds.x) == repr(np.array([[1., 0.],
                                       [0., 1.],
                                       [0., 1.],
                                       [1., 0.]]))


if __name__ == '__main__':
    unittest.main()
