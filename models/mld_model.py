from utils.structdict import StructDict, SortedStructDict

import sympy as sp
import numpy as np

import inspect
import functools

def append_call_args(func):
    def wrapper(self, *args, **kwargs):
        named_args = inspect.getfullargspec(func).args
        return func(self, *args, **kwargs, named_args=named_args)
    wrapper.__signature__ = inspect.signature(func)
    return wrapper

class MldModel(SortedStructDict):

    @append_call_args
    def __init__(self, system_matrices=None, A=None, B=None, B1=None, B2=None, B3=None, B4=None, b5=None, E1=None,
                 E2=None, E3=None, E4=None, E5=None, d=None, **kwargs):

        if system_matrices is None:
            system_matrices = set(kwargs.get('named_args')) - {'self', 'system_matrices'}
            for sys_matrix in system_matrices:
                self.__setattr__(sys_matrix, locals().get(sys_matrix))
        else:
            try:
                for key, value in system_matrices.items():
                    self.__setattr__(key, value)
            except AttributeError:
                raise TypeError("argument system_matrices must be dictionary like")
                
        super(MldModel, self).__init__()

if __name__ == '__main__':
    mld = MldModel(A=1)


