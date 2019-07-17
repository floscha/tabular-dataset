from typing import Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tabular_dataset.transformations.decorator import transformation


UNK_TOKEN = '<UNK>'


@transformation
def impute(df, columns: list, method: Optional[str] = None, fit: bool = True,
           impute_values: Optional[list] = None):
    if fit:
        if not method:
            raise ValueError("'method' has to be specified when fitting")
        if impute_values:
            raise ValueError("'impute value' argument cannot be used when " +
                             "fitting")
        if method == 'unk':
            impute_values.append(UNK_TOKEN)
        elif method == 'mode':
            impute_values.append(df.mode().iloc[0, :])
        else:
            raise ValueError("Method not supported")
    else:
        if method:
            pass  # TODO: Print warning instead
            # raise ValueError("'method' argument cannot be used when fitting")
        if not impute_values:
            raise ValueError("'impute value' has to be specified when fitting")

    return df[columns].fillna(impute_values[0])


@transformation
def encode(df: pd.DataFrame, columns: list, encoders: dict, fit: bool = True) \
        -> pd.DataFrame:
    for column_name in columns:
        if fit:
            encoder = LabelEncoder()
            # FIXME Only convert columns with object type to str?
            df[column_name] = (encoder.fit_transform(df[column_name]
                                                     .values.astype(str)))
            encoders[column_name] = encoder
        else:
            encoder = encoders[column_name]
            df[column_name] = (encoder.transform(df[column_name]
                                                 .values.astype(str)))
    return df


@transformation
def hash(df: pd.DataFrame, columns: list, bins: int) -> pd.DataFrame:
    for column_name in columns:
        df[column_name] = df[column_name] % bins
    return df


@transformation
def one_hot(df: pd.DataFrame, columns: list, encoders: dict,
            fit: bool = True) -> pd.DataFrame:
    encoded_columns = list()
    for column_name in columns:
        values = df[column_name].values.reshape(-1, 1)
        if fit:
            ohe = OneHotEncoder(categories='auto', sparse=False)
            ohe.fit(values)
            encoders[column_name] = ohe
        else:
            ohe = encoders[column_name]
        new_column_names = list(ohe.get_feature_names(column_name))
        encoded_columns.append(pd.DataFrame(ohe.transform(values),
                                            columns=new_column_names,
                                            # Make sure to retain the old index
                                            index=df[column_name].index))
    # TODO Drop the first column per feature
    df = pd.concat(encoded_columns, axis=1)
    return df
