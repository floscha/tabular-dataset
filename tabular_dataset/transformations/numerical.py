from typing import List, Optional

from sklearn import preprocessing

from tabular_dataset.transformations.decorator import transformation


@transformation
def impute(df, columns: list, method: Optional[str] = None, fit: bool = True,
           impute_values: Optional[List[float]] = None):
    if fit:
        if not method:
            raise ValueError("'method' has to be specified when fitting")
        if impute_values:
            raise ValueError("'impute value' argument cannot be used when " +
                             "fitting")

        if method == 'median':
            impute_values.extend(df.median())
        elif method == 'mean':
            impute_values.extend(df.mean())
        elif method == 'zero':
            impute_values.append(0)
        else:
            raise ValueError("Method not supported")
    else:
        if method:
            pass  # TODO: Print warning instead
            # raise ValueError("'method' argument cannot be used when fitting")
        if not impute_values:
            raise ValueError("'impute value' has to be specified when fitting")

    return df[columns].fillna(impute_values[0] if len(impute_values) == 1
                              else impute_values)


@transformation
def normalize(df, columns: list, scalers, method: str, fit: bool):
    scaler = scalers[0]
    if fit:
        if scaler is not None:
            raise ValueError("Values have already been normalized?!")

        if method == 'minmax':
            scaler = preprocessing.MinMaxScaler()
            # TODO Save scaler
            df[columns] = scaler.fit_transform(df[columns])
            scalers[0] = scaler
        else:
            raise ValueError("Method not supported")
    else:
        if scaler is None:
            raise ValueError("Scaler has to be fit first")
        df[columns] = scaler.transform(df[columns])

    return df
