import inspect
from inspect import Parameter
from collections import namedtuple, OrderedDict
import types


class ParNotSetType:
    __slots__ = ()

    def __bool__(self):
        return False

    def __repr__(self):
        return "ParNotSet"


ParNotSet = ParNotSetType()

_FSpecNamedTup = namedtuple('f_spec', ['signature', 'arg_spec', 'pos_only_params', 'pos_or_kw_params', 'kw_only_params',
                                       'all_kw_params', 'all_kw_default', 'type'])


def get_cached_func_spec(func, bypass_cache=False, clear_cache=False):
    try:
        f_spec = func._f_spec
        if f_spec is not None and not(bypass_cache or clear_cache):
            return func._f_spec  # use cached _f_spec
    except AttributeError:
        pass

    f_signature = inspect.signature(func)

    f_args_spec_dict = OrderedDict(inspect.getfullargspec(func)._asdict())
    if isinstance(func, types.MethodType):
        f_args_spec_dict['args'] = f_args_spec_dict['args'][1:]
    f_args_spec = inspect.FullArgSpec(**f_args_spec_dict)

    f_pos_only_params = [param_name for param_name, param in f_signature.parameters.items() if
                         param.kind is Parameter.POSITIONAL_ONLY]
    f_pos_or_kw_params = [param_name for param_name, param in f_signature.parameters.items() if
                          param.kind is Parameter.POSITIONAL_OR_KEYWORD]
    f_kw_only_params = [param_name for param_name, param in f_signature.parameters.items() if
                        param.kind is Parameter.KEYWORD_ONLY]
    f_all_kw_params = f_pos_or_kw_params + f_kw_only_params
    f_all_kw_default = {param_name: param.default for param_name, param in f_signature.parameters.items() if
                        param.default is not Parameter.empty}
    f_type = type(func)

    f_spec = _FSpecNamedTup(f_signature, f_args_spec, f_pos_only_params, f_pos_or_kw_params, f_kw_only_params,
                            f_all_kw_params, f_all_kw_default, f_type)

    cache_f_spec = f_spec if not clear_cache else None
    if (not bypass_cache) or clear_cache:
        if isinstance(func, types.MethodType):
            func.__func__._f_spec = cache_f_spec  # cache f_spec as function attribute
        else:
            func._f_spec = cache_f_spec

    return f_spec


def make_function(arg_spec, name='func', body='pass', globals=None, locals=None):
    arg_spec_str = inspect.formatargspec(*arg_spec)
    globals = globals if globals is not None else {}
    locals = locals if locals is not None else {}
    exec(f"def {name}{arg_spec_str}: {body}", globals, locals)
    func = locals[name]
    return func


_ArgsKwargs = namedtuple('ArgsKwargs', ['pos_only_args', 'var_args', 'all_kw_args'])
def make_args_kwargs_getter(func, f_spec=None):
    f_spec = f_spec if f_spec is not None else get_cached_func_spec(func, bypass_cache=True)

    if f_spec.pos_only_params:
        pos_only_args = f"tuple([all_kw_args.pop(arg) for arg in {f_spec.pos_only_params}])"
    else:
        pos_only_args = "()"

    if f_spec.arg_spec.varargs:
        var_args = f"all_kw_args.pop('{f_spec.arg_spec.varargs}')"
    else:
        var_args = "()"

    if f_spec.arg_spec.varkw:
        process_varkw = f"all_kw_args.update(all_kw_args.pop('{f_spec.arg_spec.varkw}'))"
    else:
        process_varkw = ""

    body = f"""
        all_kw_args = locals()
        pos_only_args = {pos_only_args}
        var_args = {var_args}
        {process_varkw}
        return _ArgsKwargs(pos_only_args, var_args, all_kw_args)
    """
    return make_function(f_spec.arg_spec, name=func.__name__ + "_args_kwargs_getter", body=body,
                         globals={'_ArgsKwargs': _ArgsKwargs})
