from types import MethodType, BuiltinMethodType
from reprlib import recursive_repr
from sortedcontainers import SortedDict
from collections import OrderedDict, namedtuple as NamedTuple
from copy import deepcopy as _deepcopy
from abc import ABCMeta

from contextlib import contextmanager
import sys


@contextmanager
def _temp_mod_numpy_print_ops(np_print_threshold=None):
    np_mod = sys.modules.get('numpy')
    cur_np_threshold = None
    if np_mod:
        cur_np_threshold = np_mod.get_printoptions()['threshold']
        np_mod.set_printoptions(threshold=np_print_threshold)
    try:
        yield np_mod
    finally:
        if np_mod:
            np_mod.set_printoptions(threshold=cur_np_threshold)


def struct_repr(data, type_name=None, sort=False, np_print_threshold=20, align_values=True, align_padding_width=0,
                value_format_str=None, repr_format_str=None):
    if not isinstance(data, dict):
        raise TypeError("Data must be dictionary like")

    _key = data._key if hasattr(data, '_key') else None
    key_arg = '' if _key is None else '_key = {0!r},\n'.format(_key)

    align_padding_str = ' ' * align_padding_width
    value_format_str = value_format_str or '{value!r}'
    repr_format_str = (
        (repr_format_str or '{type_name}({{\n{key_arg}{items}}})') if data else '{type_name}({{{key_arg}{items}}})')

    item_format_string = (
        "{{key!r:<{{width}}}}: {align_padding_str}{value_format_str}".format(align_padding_str=align_padding_str,
                                                                             value_format_str=value_format_str))

    item_format = item_format_string.format
    filler_calc = lambda key_width: ''.join(['\n', ' ' * (key_width + 2 + align_padding_width)])
    keys = sorted(data.keys(), key=_key) if sort else list(data.keys())
    key_widths = [len(repr(key)) for key in keys]
    with _temp_mod_numpy_print_ops(np_print_threshold=np_print_threshold):  # temporarily modify numpy print threshold
        if align_values:
            width = max(key_widths)
            fill = filler_calc(width)
            items = ',\n'.join(item_format(key=key, value=data[key], width=width).replace('\n', fill) for key in keys)
        else:
            width = 0
            items = ',\n'.join(
                item_format(key=key, value=data[key], width=width).replace('\n', filler_calc(key_width)) for
                key, key_width in zip(keys, key_widths))

    type_name = type_name if type_name is not None else type(data).__name__
    repr_format = repr_format_str.format
    return repr_format(type_name=type_name, key_arg=key_arg, items=items)


class StructDictMeta(ABCMeta):
    _base_dict_methods = ['__setitem__', '__getitem__', '__delitem__', 'update', 'setdefault']
    _base_dict_method_map = {"".join(['_base_dict_', method.strip('_')]): method for method in _base_dict_methods}

    def __new__(cls, name, bases, _dict, **kwargs):
        kls = super().__new__(cls, name, bases, _dict, **kwargs)
        mro = kls.mro()
        # find 1 based reversed index of StructDictMixin - i.e. first instance of class with type==StructDictMeta
        index, _ = next(filter(
            lambda index_class: type(index_class[1]) == StructDictMeta,
            enumerate(reversed(mro))))
        # this gives the base_dict as follows:
        _base_dict = mro[-index]
        setattr(kls, '_base_dict', _base_dict)
        # extract cached method references
        if issubclass(_base_dict, dict):
            for method, _based_dict_method in cls._base_dict_method_map.items():
                setattr(kls, method, getattr(_base_dict, _based_dict_method))

        _internal_name_set = set(kls._internal_names_set) if hasattr(kls, '_internal_names_set') else set()
        kls._internal_names_set = _internal_name_set.union(dir(kls))
        return kls


class StructDictMixin(metaclass=StructDictMeta):
    __internal_names = []
    _internal_names_set = set(__internal_names)

    def __init__(self, *args, **kwargs):
        super(StructDictMixin, self).__init__(*args, **kwargs)
        self._check_invalid_keys()

    def _check_invalid_keys(self):
        invalid_keys = self._internal_names_set.intersection(self.keys())
        if invalid_keys:
            for key in invalid_keys:
                self._base_dict_delitem(key)
            raise ValueError(
                "Cannot add items to struct dict with keys contained in '_internal_names_set': '{}'".format(
                    invalid_keys))

    def __setitem__(self, key, value):
        self._base_dict_setitem(key, value)
        if key in self._internal_names_set:
            self._check_invalid_keys()

    @property
    def base_dict(self):
        """Return the base_dict of the struct_dict."""
        return self._base_dict

    def update(self, *args, **kwargs):
        self._base_dict_update(*args, **kwargs)
        self._check_invalid_keys()

    def setdefault(self, key, default=None):
        self._base_dict_setdefault(key, default)
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
                    "Cannot add item:'{}' to struct via __setattr__, identifier is an object method.".format(key))
            else:
                return object.__setattr__(self, key, value)

        if self._internal_names_set and key in self._internal_names_set:
            object.__setattr__(self, key, value)
        else:
            self._base_dict_setitem(key, value)

    def __getattr__(self, key):
        try:
            return self._base_dict_getitem(key)
        except KeyError:
            raise AttributeError("'{0}' object has no attribute '{1}'".format(self.__class__.__name__, key))

    def __dir__(self):
        orig_dir = set(dir(type(self)))
        __dict__keys = set(self.__dict__.keys())
        additions = {key for key in list(self.keys())[:100] if isinstance(key, str)}
        rv = orig_dir | __dict__keys | additions
        return sorted(rv)

    @recursive_repr()
    def __repr__(self):
        """Return string representation of struct dict.
        """
        return struct_repr(self, sort=False)

    def __reduce__(self):
        """Support for pickle.

        The tricks played with caching references in
        :func:`StructDictMixin.__new__` confuse pickle so customize the reducer.

        """
        if hasattr(self, '_key'):
            args = (self._key, dict(self))
        else:
            args = (dict(self),)
        return (self.__class__, args)

    def copy(self):
        if hasattr(self, '_key'):
            return self.__class__(self._key, self)
        else:
            return self.__class__(self)

    __copy__ = copy

    def deepcopy(self, memo=None):
        if hasattr(self, '_key'):
            return self.__class__(self._key, _deepcopy(dict(self), memo=memo))
        else:
            return self.__class__(_deepcopy(dict(self), memo=memo))

    __deepcopy__ = deepcopy

    def get_sub_base_dict(self, keys):
        try:
            return self._base_dict([(key, self[key]) for key in keys])
        except KeyError as ke:
            raise KeyError(f"Invalid key in keys: '{ke.args[0]}'")

    def get_sub_list(self, keys):
        try:
            return [self[key] for key in keys]
        except KeyError as ke:
            raise KeyError(f"Invalid key in keys: '{ke.args[0]}'")

    def get_sub_struct(self, keys):
        return self.__class__(self.get_sub_base_dict(keys))

    @classmethod
    def sub_struct_fromdict(cls, dict_, keys):
        return cls(cls.get_sub_base_dict(dict_, keys))

    def as_base_dict(self):
        """Return a new base_dict of the struct_dict which maps item names to their values."""
        return self._base_dict(self)


class StructDict(StructDictMixin, dict):
    @recursive_repr()
    def __repr__(self):
        """Return string representation of StructDict
        """
        return struct_repr(self, sort=True)


class OrderedStructDict(StructDictMixin, OrderedDict):
    pass


class SortedStructDict(StructDictMixin, SortedDict):
    # extract all internal names of SortedDict
    __internal_names = list(SortedDict(callable).__dict__.keys())
    _internal_names_set = StructDictMixin._internal_names_set.union(__internal_names)


class FrozenStructDict(StructDict):
    def __setitem__(self, key, value):
        raise TypeError("'{0}' object does not support item assignment".format(self.__class__.__name__))

    def update(self, *args, **kwargs):
        raise TypeError("'{0}' object does not support update".format(self.__class__.__name__))


if __name__ == '__main__':
    st = StructDict(a=1, b=2)
    sst = SortedStructDict(b=1, a=2)
