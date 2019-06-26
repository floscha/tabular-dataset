import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tabular_dataset.transformations.decorator import transformation


UNK_TOKEN = '<UNK>'


@transformation
def impute(df: pd.DataFrame, columns: list, method: str):
  if method == 'unk':
    fill_values = UNK_TOKEN
  elif method == 'mode':
    fill_values = df.mode().iloc[0, :]
  else:
    raise ValueError("Method not supported")

  return df.fillna(fill_values)


@transformation
def encode(df: pd.DataFrame, columns: list, encoders: dict) -> pd.DataFrame:
  for column_name in columns:
    encoder = LabelEncoder()
    # FIXME Only convert columns with object type to str?
    df[column_name] = encoder.fit_transform(df[column_name].values.astype(str))
    encoders[column_name] = encoder
  return df


@transformation
def hash(df: pd.DataFrame, columns: list, bins: int) -> pd.DataFrame:
  for column_name in columns:
    df[column_name] = df[column_name] % bins
  return df


@transformation
def one_hot(df: pd.DataFrame, columns: list) -> pd.DataFrame:
  old_column_names = columns
  ohe = OneHotEncoder(categories='auto', sparse=False)
  ohe.fit(df[old_column_names])
  new_column_names = list(ohe.get_feature_names(old_column_names))
  df = df.join(
    pd.DataFrame(ohe.transform(df[old_column_names]), columns=new_column_names)
  )
  # TODO Drop the first column per feature
  return df.drop(old_column_names, axis=1)
