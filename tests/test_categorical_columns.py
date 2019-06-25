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


def test_encode_with_hashing():
  df = pd.DataFrame({'C': range(5)})

  tds = TabularDataset(df, categorical_columns=['C'])
  tds.categorical.encode()
  tds.categorical.hash(bins=3)

  assert repr(tds.x) == repr(np.array([0, 1, 2, 0, 1]).reshape(-1, 1))


def test_encode_one_hot():
  df = get_test_df()

  tds = TabularDataset(df, categorical_columns=['C'])
  tds.categorical.encode()
  tds.categorical.one_hot()

  assert repr(tds.x) == repr(np.array([[1., 0.],
                                       [0., 1.],
                                       [0., 1.],
                                       [1., 0.]]))


def test_encode_one_hot_with_hashing():
  df = pd.DataFrame({'C': range(5)})

  tds = TabularDataset(df, categorical_columns=['C'])
  tds.categorical.encode()
  tds.categorical.hash(bins=3)
  tds.categorical.one_hot()

  assert repr(tds.x) == repr(np.array([[1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., 1.],
                                       [1., 0., 0.],
                                       [0., 1., 0.]]))


def test_impute_with_unk_token():
  df = get_test_df()

  tds = TabularDataset(df, categorical_columns=['B'])
  tds.categorical.impute()

  assert repr(tds.x) == repr(np.array([[0.], [1.], [0.], ['<UNK>']],
                                      dtype='object'))


def test_impute_with_mode():
  df = get_test_df()

  tds = TabularDataset(df, categorical_columns=['B'])
  tds.categorical.impute(method='mode')

  assert repr(tds.x) == repr(np.array([[0.], [1.], [0.], [0.]]))


def test_chaining_all_transformations():
  df = get_test_df()

  tds = TabularDataset(df, categorical_columns=['B'])
  tds.categorical.impute()
  tds.categorical.encode()

  assert repr(tds.x) == repr(np.array([[0], [1], [0], [2]]))


if __name__ == '__main__':
    unittest.main()
