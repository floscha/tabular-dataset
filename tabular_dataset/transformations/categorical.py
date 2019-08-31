from typing import Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tabular_dataset.transformations.common import add_imputed_columns
from tabular_dataset.transformations.decorator import transformation


NAN_TOKEN = '<NaN>'
UNK_TOKEN = '<UNK>'


@transformation
def impute(df: pd.DataFrame, columns: List[str], impute_values: list,
           method: Optional[str] = None, fit: bool = True,
           add_columns: bool = False) -> pd.DataFrame:
    if fit:
        if not method:
            raise ValueError("'method' has to be specified when fitting")
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

    if add_columns:
        add_imputed_columns(df, columns)

    df[columns] = df[columns].fillna(impute_values[0])

    return df


@transformation
def encode(df: pd.DataFrame, columns: List[str],
           encoders: Dict[str, LabelEncoder], fit: bool = True) \
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
def hash(df: pd.DataFrame, columns: List[str], bins: int) -> pd.DataFrame:
    for column_name in columns:
        df[column_name] = df[column_name] % bins
    return df


@transformation
def one_hot(df: pd.DataFrame, columns: List[str],
            encoders: Dict[str, OneHotEncoder], fit: bool = True,
            drop_first: bool = False) -> pd.DataFrame:
    encoded_columns = list()
    for column_name in columns:
        values = df[column_name].values.reshape(-1, 1)
        if fit:
            ohe = OneHotEncoder(categories='auto', sparse=False)
            ohe.fit(values)
            encoders[column_name] = ohe
        else:
            ohe = encoders[column_name]
        new_column_names = list(ohe.get_feature_names([column_name]))
        new_columns = pd.DataFrame(ohe.transform(values),
                                   columns=new_column_names,
                                   # Make sure to retain the old index
                                   index=df[column_name].index)
        columnar_offset = 1 if drop_first else 0
        encoded_columns.append(new_columns.iloc[:, columnar_offset:])
    return pd.concat(encoded_columns, axis=1)


@transformation
def counts(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column_name in columns:
        counts = df[column_name].value_counts()
        df[column_name + '_count'] = df[column_name].map(counts)
    return df


@transformation
def frequencies(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column_name in columns:
        counts = df[column_name].value_counts()
        freqs = counts / len(df)
        df[column_name + '_freq'] = df[column_name].map(freqs)
    return df
