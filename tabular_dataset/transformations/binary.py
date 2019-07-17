from tabular_dataset.transformations.decorator import transformation


@transformation
def impute(df, columns: list):
    return df[columns].fillna(0)


@transformation
def encode(df, columns: list):
    for column_name in columns:
        column_values = df[column_name].dropna().unique()
        if len(column_values) != 2:
            raise ValueError("Binary variables need to have EXACTLY two " +
                             "different values (except for NaN values)")
        smaller_value, bigger_value = sorted(column_values)
        mapping = {smaller_value: -1, bigger_value: 1}
        df[column_name] = df[column_name].map(mapping)
    return df
