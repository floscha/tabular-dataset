from typing import List, Optional, Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_categorical_columns(columns, one_hot: bool = False,
                               hash_bins: Optional[int] = None):
  for column_name in columns.column_names:
    encoder = LabelEncoder()
    # FIXME Only convert columns with object type to str?
    encoded_values = encoder.fit_transform(
      columns.ds.df[columns.column_names].values.astype(str).ravel())
    columns.ds.df[column_name] = (encoded_values % hash_bins if hash_bins
                                  else encoded_values)
    columns._encoders[column_name] = encoder

  if one_hot:
    old_column_names = columns.column_names
    ohe = OneHotEncoder(categories='auto', sparse=False)
    new_column_names = list(ohe.fit(columns.ds.df[old_column_names])
                            .get_feature_names(old_column_names))
    columns.ds.df = columns.ds.df.join(pd.DataFrame(ohe.transform(columns.ds.df[old_column_names]),
                                                    columns=new_column_names))
    # TODO Drop the first column per feature
    columns.ds.df = columns.ds.df.drop(old_column_names, axis=1)
    columns.column_names = new_column_names
