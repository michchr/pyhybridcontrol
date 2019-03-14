import inspect
from inspect import Parameter
from collections import namedtuple, OrderedDict
import types
import functools
import importlib


class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ParNotSetType(metaclass=SingletonMeta):
    __slots__ = ()

    def __bool__(self):
        return False

    def __repr__(self):
        return "ParNotSet"


ParNotSet = ParNotSetType()

_FSpecNamedTup = namedtuple('f_spec', ['signature', 'arg_spec', 'pos_only_params', 'pos_or_kw_params', 'kw_only_params',
                                       'all_kw_params', 'all_kw_default', 'type'])


def get_cached_func_spec(func, bypass_cache=False, reset_cache=False, clear_cache=False):
    try:
        f_spec = func._f_spec
        if f_spec is not None and not any((bypass_cache, reset_cache, clear_cache)):
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
    if (not bypass_cache) or reset_cache or clear_cache:
        if isinstance(func, types.MethodType):
            func.__func__._f_spec = cache_f_spec  # cache f_spec as function attribute
        else:
            func._f_spec = cache_f_spec

    return f_spec


def make_function(arg_spec, name='func', body='pass', global_ns=None, local_ns=None, formatannotation=None):
    if formatannotation:
        arg_spec_str = inspect.formatargspec(*arg_spec, formatannotation=formatannotation)
    else:
        arg_spec_str = inspect.formatargspec(*arg_spec)

    global_ns = global_ns if global_ns is not None else {}
    local_ns = local_ns if local_ns is not None else {}
    exec(f"def {name}{arg_spec_str}: {body}", global_ns, local_ns)
    func = local_ns[name]
    return func


_ArgsKwargs = namedtuple('ArgsKwargs', ['pos_only_args', 'var_args', 'all_kw_args', 'var_kwargs'])


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
        var_kwargs = f"all_kw_args.pop('{f_spec.arg_spec.varkw}')"
        process_kwargs = f"all_kw_args.update(var_kwargs)"
    else:
        var_kwargs = "{}"
        process_kwargs = ""

    body = f"""
        all_kw_args = locals()
        #pos_only_args = {pos_only_args}
        #var_args = {var_args}
        var_kwargs = {var_kwargs}
        {process_kwargs}
        return _ArgsKwargs({pos_only_args}, {var_args}, all_kw_args, var_kwargs)
    """

    global_ns = {'_ArgsKwargs': _ArgsKwargs}
    global_ns.update(func.__globals__)

    for klass in func.__annotations__.values():
        if klass.__module__ not in ('builtins', func.__module__):
            package = inspect.getmodule(klass).__package__
            if package:
                global_ns[package] = importlib.import_module(package)
            else:
                global_ns[klass.__module__] = importlib.import_module(klass.__module__)

    formatannotation = functools.partial(inspect.formatannotation, base_module=func.__module__)

    return make_function(f_spec.arg_spec, name=func.__name__ + "_args_kwargs_getter", body=body,
                         global_ns=global_ns, formatannotation=formatannotation)
