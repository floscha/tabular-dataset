from sklearn import preprocessing


class NumericalColumns:
  def __init__(self, ds, column_names):
    self.ds = ds
    self.column_names = column_names

    self._scaler = None

  def normalize(self, method: str = 'minmax'):
    if self._scaler is not None:
      raise ValueError("Values have already been normalized?!")

    if method == 'minmax':
      self._scaler = preprocessing.MinMaxScaler()
    else:
      raise ValueError("Method not supported")

    self.ds.df[self.column_names] = self._scaler.fit_transform(
      self.ds.df[self.column_names]
    )

    return self.ds  # For fluent API

  def impute(self, method='median'):
    if method == 'median':
      impute_value = self.ds.df[self.column_names].median()
    elif method == 'mean':
      impute_value = self.ds.df[self.column_names].mean()
    elif method == 'zero':
      impute_value = 0
    else:
      raise ValueError("Method not supported")

    self.ds.df[self.column_names] = (self.ds.df[self.column_names]
                                     .fillna(impute_value))

    return self.ds  # For fluent API
