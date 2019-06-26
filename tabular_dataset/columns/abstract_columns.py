class AbstractColumns:
  def __init__(self, ds, column_names):
    self.ds = ds
    self.column_names = column_names

    self.lineage = []

  def transform(self):
    df = self.ds.df[self.column_names].copy()

    for transformation_fn in self.lineage:
      df = transformation_fn(df)

    return df
