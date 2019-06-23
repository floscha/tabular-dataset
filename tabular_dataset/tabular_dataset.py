import pandas as pd

from tabular_dataset.numerical_columns import NumericalColumns


class TabularDataset:
  def __init__(self, data, numerical_columns):
    self.df = data

    self.numerical = NumericalColumns(self, numerical_columns or [])
