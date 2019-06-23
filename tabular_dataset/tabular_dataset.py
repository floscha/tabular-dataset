import pandas as pd

from tabular_dataset.binary_columns import BinaryColumns
from tabular_dataset.categorical_columns import CategoricalColumn
from tabular_dataset.numerical_columns import NumericalColumns


class TabularDataset:
  def __init__(self, data, numerical_columns=None, binary_columns=None,
               categorical_columns=None):
    self.df = data

    self.numerical = NumericalColumns(self, numerical_columns or [])
    self.binary = BinaryColumns(self, binary_columns or [])
    self.categorical = CategoricalColumn(self, categorical_columns or [])
