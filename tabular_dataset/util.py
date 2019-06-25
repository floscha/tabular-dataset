from typing import List, Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_categorical_columns(columns, one_hot: bool = False):
  all_new_column_names = []
  for column_name in columns.column_names:
    columns._encoders[column_name] = (OneHotEncoder(sparse=False) if one_hot
                                      else LabelEncoder())
    if one_hot:
      # FIXME Only convert columns with object type to str?
      encoded_values = columns._encoders[column_name].fit_transform(
        columns.ds.df[columns.column_names].values.astype(str).reshape(-1, 1))
      # TODO Drop the first column per feature
      new_column_names = (columns._encoders[column_name]
                          .get_feature_names([column_name]))
      columns.ds.df = columns.ds.df.drop(column_name, axis=1)
      for i, new_column in enumerate(new_column_names):
        columns.ds.df[new_column] = encoded_values[:, i]
      all_new_column_names.extend(new_column_names)
    else:
      # FIXME Only convert columns with object type to str?
      encoded_values = columns._encoders[column_name].fit_transform(
        columns.ds.df[columns.column_names].values.astype(str).ravel())
      columns.ds.df[column_name] = encoded_values

  if one_hot:
    columns.column_names = all_new_column_names
