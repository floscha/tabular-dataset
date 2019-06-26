class AllColumns:
    def __init__(self, ds):
        self.ds = ds

    @property
    def column_names(self):
        return (self.ds.numerical.column_names +
                self.ds.binary.column_names +
                self.ds.categorical.column_names)
