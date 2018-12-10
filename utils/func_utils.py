import inspect
from collections import namedtuple as NamedTuple
import types

_f_spec_named_tuple = NamedTuple('f_spec', ['signature', 'arg_spec', 'kwargs_default'])


def get_cached_func_spec(func):
    try:
        return func._f_spec  # use cached _f_spec
    except AttributeError:
        f_signature = inspect.signature(func)
        f_args_spec = inspect.getfullargspec(func)
        f_kwargs_default = {param_name: param.default for param_name, param in f_signature.parameters.items() if
                            param.default is not inspect.Parameter.empty}

        f_spec = _f_spec_named_tuple(f_signature, f_args_spec, f_kwargs_default)
        if isinstance(func, types.MethodType):
            func.__func__._f_spec = f_spec  # cache f_spec as function attribute
        else:
            func._f_spec = f_spec
        return f_spec


