from typing import List, Optional, Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_categorical_columns(columns_obj, columns):
  for column_name in columns:
    encoder = LabelEncoder()
    # FIXME Only convert columns with object type to str?
    columns_obj.ds.df[column_name] = encoder.fit_transform(
      columns_obj.ds.df[column_name].values.astype(str))
    columns_obj._encoders[column_name] = encoder


def hash_categorical_columns(columns_obj, columns, bins: int):
  for column_name in columns:
    columns_obj.ds.df[column_name] = columns_obj.ds.df[column_name] % bins


def one_hot_categorical_columns(columns_obj, columns):
  old_column_names = columns
  ohe = OneHotEncoder(categories='auto', sparse=False)
  ohe.fit(columns_obj.ds.df[old_column_names])
  new_column_names = list(ohe.get_feature_names(old_column_names))
  columns_obj.ds.df = columns_obj.ds.df.join(
    pd.DataFrame(ohe.transform(columns_obj.ds.df[old_column_names]),
                 columns=new_column_names)
  )
  # TODO Drop the first column per feature
  columns_obj.ds.df = columns_obj.ds.df.drop(old_column_names, axis=1)
  columns_obj.column_names = ([cn for cn in columns_obj.column_names
                               if cn not in old_column_names]
                              + new_column_names)
