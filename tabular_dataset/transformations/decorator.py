from functools import partial


def transformation(fn):
    def wrapper(*args, **kwargs):
        return partial(fn, *args, **kwargs)
    return wrapper
