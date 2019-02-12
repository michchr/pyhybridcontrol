import os
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'True'
import wrapt
from types import  MethodType, BuiltinMethodType

def A(A):
    b = A
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        print (b)
        return wrapped(*args ,**kwargs)
    return wrapper

@A(1)
def test(a,b,c=1,*args, k=1, **kwargs):
    return 1

@A(3)
def test2(a,b,c=1,*args, k=1, **kwargs):
    return 1


class St2(dict):
    __internal_names = []
    _internal_names_set = set(__internal_names)

    def __init__(self, *args, **kwargs):
        super(St2, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self._check_invalid_keys()



    def _check_invalid_keys(self):
        invalid_keys = self._internal_names_set.intersection(self.keys())
        if invalid_keys:
            for key in invalid_keys:
                self.__delitem__(key)
            raise ValueError(
                f"Cannot add items to struct dict with keys contained in '_internal_names_set': '{invalid_keys}'")

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        if key in self._internal_names_set:
            self._check_invalid_keys()

    def __setattr__(self, key, value):
        try:
            attr = object.__getattribute__(self, key)
        except AttributeError:
            pass
        else:
            if isinstance(attr, (MethodType, BuiltinMethodType)):
                raise ValueError(
                    f"Cannot modify or add item:'{key}' to struct, '{key}' is a {self.__class__} object method.")
            else:
                return object.__setattr__(self, key, value)

        if key in self._internal_names_set:
            return object.__setattr__(self, key, value)
        else:
            self.__setitem__(key, value)



class A:
    __slots__ = ('a','b','c')


class B(A):
    __slots__ = ('d','e','f')
