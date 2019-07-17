import inspect


class AbstractColumns:
    def __init__(self, ds, column_names):
        self.ds = ds
        self.column_names = column_names

        self.lineage = []

    def __bool__(self):
        return len(self) > 0

    def __len__(self):
        return len(self.column_names)

    def transform(self, test: bool = False):
        df = self.ds.test_df if test else self.ds.df
        df = df[self.column_names].copy()

        for transformation_fn in self.lineage:
            parameters = inspect.signature(transformation_fn).parameters
            df = (transformation_fn(df, fit=not test) if 'fit' in parameters
                  else transformation_fn(df))

        return df
