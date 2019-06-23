from typing import List, Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# class MultiColumnLabelEncoder:
#   """
#
#   Code mostly taken from https://stackoverflow.com/a/30267328
#   """
#   def __init__(self, columns = None):
#     self.columns = columns # array of column names to encode
#
#   def fit(self, X, y=None):
#     return self # not relevant here
#
#   def transform(self, X):
#     """Transforms columns of X specified in self.columns using LabelEncoder().
#
#     If no columns specified, transforms all columns in X.
#     """
#     output = X.copy()
#     if self.columns is not None:
#       for col in self.columns:
#         output[col] = LabelEncoder().fit_transform(output[col])
#     else:
#       for colname, col in output.iteritems():
#         output[colname] = LabelEncoder().fit_transform(col)
#     return output
#
#   def fit_transform(self, X, y=None):
#     return self.fit(X, y).transform(X)


def encode_categorical_columns(columns, one_hot: bool = False):
  all_new_column_names = []
  for column_name in columns.column_names:
    columns._encoders[column_name] = (OneHotEncoder(sparse=False) if one_hot
                                      else LabelEncoder())
    if one_hot:
      encoded_values = columns._encoders[column_name].fit_transform(
        columns.ds.df[columns.column_names].values.reshape(-1, 1))
      # TODO Drop the first column per feature
      new_column_names = (columns._encoders[column_name]
                          .get_feature_names(column_name))
      columns.ds.df = columns.ds.df.drop(column_name, axis=1)
      for i, new_column in enumerate(new_column_names):
        columns.ds.df[new_column] = encoded_values[:, i]
      all_new_column_names.extend(new_column_names)
    else:
      encoded_values = columns._encoders[column_name].fit_transform(
        columns.ds.df[columns.column_names].values.ravel())
      columns.ds.df[column_name] = encoded_values

  if one_hot:
    columns.column_names = all_new_column_names
