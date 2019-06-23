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


def test_encode_one_hot():
  df = get_test_df()

  tds = TabularDataset(df, target_column='target')
  tds.target.encode(one_hot=True)

  assert repr(tds.y) == repr(np.array([[1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., 1.],
                                       [1., 0., 0.]]))


if __name__ == '__main__':
    unittest.main()
