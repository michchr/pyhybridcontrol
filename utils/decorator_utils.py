import functools
import inspect

import wrapt

from utils.func_utils import get_cached_func_spec, ParNotSet


def process_method_args_decor(*processors):
    def wrapper_up(func):
        f_spec = get_cached_func_spec(func.__get__(ParNotSet)) #ensure f_spec treats func as a method
        @wrapt.decorator(adapter=func)
        def wrapper(wrapped, self, args_in, kwargs_in):
            if kwargs_in.pop('_disable_process_args', False):
                return wrapped(*args_in, **kwargs_in)
            if args_in:
                kw_update = {param_name: value for value, param_name in
                             zip(args_in[len(f_spec.pos_only_params):], f_spec.pos_or_kw_params)}
                args = args_in[:-len(kw_update)]
                kwargs_in.update(kw_update)
            else:
                args = ()

            kwargs = dict(f_spec.all_kw_default, **kwargs_in)

            for processor in processors:
                p_func = getattr(self, processor)
                p_spec = get_cached_func_spec(p_func)
                if p_spec.arg_spec.varkw:
                    p_func(f_kwargs=kwargs, *args, **kwargs)
                else:
                    required_kwargs = set(kwargs).intersection(p_spec.all_kw_default)
                    p_func(f_kwargs=kwargs, *args, **{key: kwargs[key] for key in required_kwargs})
            return wrapped(*args, **kwargs)

        return wrapper(func)

    return wrapper_up


def cache_hashable_args(maxsize=128, typed=False):
    def lru_wrapper(func):
        lru_wrapped = functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        @wrapt.decorator(adapter=func)
        def wrapper(func, instance, args, kwargs):
            return func(*args, **kwargs)

        return wrapper(lru_wrapped)

    return lru_wrapper


if __name__ == '__main__':
    def test(a, b=1):
        """This is test"""
        return a + b


    test_dec = cache_hashable_args()(test)
