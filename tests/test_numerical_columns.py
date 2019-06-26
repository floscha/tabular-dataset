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


def test_impute_with_median():
  df = get_test_df()

  tds = TabularDataset(df, numerical_columns=['A'])
  tds.numerical.impute()

  assert repr(tds.x) == repr(np.array([[1], [2], [3], [2.]]))


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


def test_fluent_api():
  df = get_test_df()

  tds = TabularDataset(df, numerical_columns=['A'])
  (tds
   .numerical.impute()
   .numerical.normalize())

  assert repr(tds.x) == repr(np.array([[0.], [0.5], [1.], [0.5]]))


if __name__ == '__main__':
    unittest.main()
