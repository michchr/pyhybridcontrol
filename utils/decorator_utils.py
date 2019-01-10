import functools
import inspect

import wrapt

from utils.func_utils import get_cached_func_spec, make_args_kwargs_getter, ParNotSet


def process_method_args_decor(*processor_funcs):
    def wrapper_up(func):
        f_spec = get_cached_func_spec(func.__get__(ParNotSet))  # ensure f_spec treats func as a method
        args_kwargs_getter = make_args_kwargs_getter(func, f_spec=f_spec)
        processor_spec_funcs = [(get_cached_func_spec(p_func), p_func) for p_func in processor_funcs]

        def process_kwargs(self, processor_spec_funcs, kwargs):
            for p_spec,p_func in processor_spec_funcs:
                if p_spec.arg_spec.varkw:
                    p_func(self, f_kwargs=kwargs, **kwargs)
                else:
                    required_kwargs = set(kwargs).intersection(p_spec.all_kw_default)
                    p_func(self, f_kwargs=kwargs, **{key: kwargs[key] for key in required_kwargs})
            return kwargs

        @wrapt.decorator(adapter=func)
        def no_posonly_vargs_wrapper(wrapped, self, args_in, kwargs_in):
            if kwargs_in.pop('_disable_process_args', False):
                return wrapped(*args_in, **kwargs_in)

            try:
                _, args, kwargs = args_kwargs_getter(*args_in, **kwargs_in)
            except TypeError:
                args, kwargs = args_in, kwargs_in
                return wrapped(*args, **kwargs)

            kwargs = process_kwargs(self, processor_spec_funcs, kwargs)

            return wrapped(*args, **kwargs)

        @wrapt.decorator(adapter=func)
        def with_posonly_vargs_wrapper(wrapped, self, args_in, kwargs_in):
            if kwargs_in.pop('_disable_process_args', False):
                return wrapped(*args_in, **kwargs_in)

            try:
                pos_only_args, var_args, kwargs = args_kwargs_getter(*args_in, **kwargs_in)
            except TypeError:
                args, kwargs = args_in, kwargs_in
                return wrapped(*args, **kwargs)

            kwargs = process_kwargs(self, processor_spec_funcs, kwargs)
            args = pos_only_args + tuple([kwargs.pop(arg) for arg in f_spec.arg_spec.args]) + var_args
            return wrapped(*args, **kwargs)

        if f_spec.arg_spec.varargs or f_spec.pos_only_params:
            return with_posonly_vargs_wrapper(func)
        else:
            return no_posonly_vargs_wrapper(func)
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
