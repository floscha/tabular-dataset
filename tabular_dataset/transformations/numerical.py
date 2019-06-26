from sklearn import preprocessing


def impute(df, columns: list, method: str):
  if method == 'median':
    impute_value = df.median()
  elif method == 'mean':
    impute_value = df.mean()
  elif method == 'zero':
    impute_value = 0
  else:
    raise ValueError("Method not supported")

  return df[columns].fillna(impute_value)


def normalize(df, columns: list, scaler, method: str):
  if scaler is not None:
    raise ValueError("Values have already been normalized?!")

  if method == 'minmax':
    scaler = preprocessing.MinMaxScaler()
  else:
    raise ValueError("Method not supported")

  df[columns] = scaler.fit_transform(df[columns])

  return df
