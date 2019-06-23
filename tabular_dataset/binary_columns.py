class BinaryColumns:
  def __init__(self, ds, column_names):
    self.ds = ds
    self.column_names = column_names

  def encode(self):
     for column_name in self.column_names:
       column_values = self.ds.df[column_name].dropna().unique()
       if len(column_values) != 2:
         raise ValueError("Binary variables need to have EXACTLY two " +
                          "different values (except for NaN values)")
       smaller_value, bigger_value = sorted(column_values)
       mapping = {smaller_value: -1, bigger_value: 1}
       self.ds.df[column_name] = self.ds.df[column_name].map(mapping)

     return self.ds  # For fluent API
