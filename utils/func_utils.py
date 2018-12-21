import inspect
from inspect import Parameter
from collections import namedtuple as NamedTuple
import types

_FSpecNamedTup = NamedTuple('f_spec',
                            ['signature', 'arg_spec', 'pos_only_params', 'pos_or_kw_params', 'kw_only_params',
                                  'all_kw_params', 'all_kw_default', 'type'])


def get_cached_func_spec(func, save_to_cache=True, bypass_cache=False):
    try:
        if not bypass_cache:
            return func._f_spec  # use cached _f_spec
    except AttributeError:
        pass

    f_signature = inspect.signature(func)
    f_args_spec = inspect.getfullargspec(func)
    f_pos_only_params = [param_name for param_name, param in f_signature.parameters.items() if
                         param.kind is Parameter.POSITIONAL_ONLY]
    f_pos_or_kw_params = [param_name for param_name, param in f_signature.parameters.items() if
                          param.kind is Parameter.POSITIONAL_OR_KEYWORD]
    f_kw_only_params = [param_name for param_name, param in f_signature.parameters.items() if
                        param.kind is Parameter.KEYWORD_ONLY]
    f_all_kw_params = f_pos_or_kw_params + f_kw_only_params
    f_all_kw_default = {param_name: param.default for param_name, param in f_signature.parameters.items() if
                        param.default is not Parameter.empty}

    f_spec = _FSpecNamedTup(f_signature, f_args_spec, f_pos_only_params, f_pos_or_kw_params, f_kw_only_params,
                            f_all_kw_params, f_all_kw_default, type(func))

    if save_to_cache:
        if isinstance(func, types.MethodType):
            func.__func__._f_spec = f_spec  # cache f_spec as function attribute
        else:
            func._f_spec = f_spec

    return f_spec
