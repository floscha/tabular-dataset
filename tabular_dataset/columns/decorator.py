from functools import partial


def transformation(fn):
    def wrapper(*args, **kwargs):
        transformation_fn = fn(*args, **kwargs)
        self = args[0]
        columns = kwargs.get('columns', self.column_names)
        self.lineage.append(partial(transformation_fn, columns=columns))
        return self.ds  # For fluent API
    return wrapper
