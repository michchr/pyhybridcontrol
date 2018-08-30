from utils.structdict import StructDict, SortedStructDictAliased

import sympy as sp
import numpy as np

import inspect
import functools


def append_named_call_args(func):
    def wrapper(self, *args, **kwargs):
        named_args = inspect.getfullargspec(func).args
        return func(self, *args, **kwargs, named_call_args=named_args)

    wrapper.__signature__ = inspect.signature(func)
    return wrapper


class MldModel:

    _internal_names = ['_data']
    _internal_names_set = set(_internal_names)

    @append_named_call_args
    def __init__(self, system_matrices=None, A=None, B=None, B1=None, B2=None, B3=None, B4=None, b5=None,
                 E1=None, E2=None, E3=None, E4=None, E5=None, d=None,
                 **kwargs):

        allowed_system_matrices = set(kwargs.get('named_call_args')) - {'self', 'system_matrices'}

        self._data = SortedStructDictAliased(dict.fromkeys(allowed_system_matrices))
        if system_matrices is None:
            for sys_matrix in allowed_system_matrices:
                self[sys_matrix] = locals().get(sys_matrix)
        else:
            try:
                for key, value in system_matrices.items():
                    if key in self._data:
                        self._data[key] = value
                    else:
                        raise ValueError("Invalid matrix name in system_matrices: {}".format(key))
            except AttributeError:
                raise TypeError("argument system_matrices must be dictionary like")

    def __setattr__(self, key, value):
        try:
            object.__getattribute__(self, key)
            return object.__setattr__(self, key, value)
        except AttributeError:
            pass

        if key in self._internal_names_set:
            super(MldModel, self).__setattr__(key, value)

        if key in self._data:
            self._data[key] = value
        else:
            super(MldModel, self).__setattr__(key, value)

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, key):
        try:
            return self._data.__getattr__(key)
        except AttributeError:
            raise AttributeError("Attribute/system matrix with name: '{}' does not exist".format(key))

    def __repr__(self):
        object_type = type(self).__name__
        return "".join([object_type, '(', self._data.__repr__(),')'])

    def __dir__(self):
        orig_dir = set(dir(type(self)))
        __dict__keys = set(self.__dict__.keys())
        additions = set(self._data.keys())
        rv = orig_dir | __dict__keys | additions
        return sorted(rv)


if __name__ == '__main__':
    mld = MldModel(A=1)
    mld2 = MldModel({'A': 1})
