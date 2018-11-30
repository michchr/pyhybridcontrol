import wrapt
import functools
import inspect
from enum import IntEnum
from collections import namedtuple as NamedTuple
import types



class _ParamRequired(IntEnum):
    TRUE = True
    FALSE = False

ParNotReq = _ParamRequired.FALSE
ParReq = _ParamRequired.TRUE

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

def process_method_args_decor(*processors):
    @wrapt.decorator
    def wrapper(func, self, args_in, kwargs_in):
        if kwargs_in.pop('_disable_process_args', False):
            return func(*args_in, **kwargs_in)

        f_spec = get_cached_func_spec(func)

        if args_in:
            kw_update = {param_name: value for value, (param_name, param) in
                         zip(args_in, f_spec.signature.parameters.items()) if
                         param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD}
            args = args_in[:-len(kw_update)]
            kwargs_in.update(kw_update)
        else:
            args = ()

        kwargs = dict(f_spec.kwargs_default, **kwargs_in)

        for processor in processors:
            p_func = getattr(self, processor)
            p_spec = get_cached_func_spec(p_func)
            if p_spec.arg_spec.varkw:
                p_func(f_kwargs=kwargs, *args, **kwargs)
            else:
                required_kwargs = set(kwargs).intersection(p_spec.kwargs_default)
                p_func(f_kwargs=kwargs, *args, **{key: kwargs[key] for key in required_kwargs})
        return func(*args, **kwargs)

    return wrapper


def cache_hashable_args(maxsize=128, typed=False):
    def lru_wrapper(func):
        lru_wrapped = functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        @wrapt.decorator(adapter=func)
        def wrapper(func, instance, args, kwargs):
            return func(*args, **kwargs)

        return wrapper(lru_wrapped)

    return lru_wrapper