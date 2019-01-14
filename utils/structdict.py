import itertools
import sys as _sys
from abc import ABCMeta, abstractmethod
from builtins import property as _property
from collections import OrderedDict as _OrderedDict
from contextlib import contextmanager
from copy import deepcopy as _deepcopy
from keyword import iskeyword as _iskeyword
from reprlib import recursive_repr as _recursive_repr
from types import MethodType as _MethodType, BuiltinMethodType as _BuiltinMethodType
from inspect import isclass as _isclass

from sortedcontainers import SortedDict as _SortedDict

from utils.func_utils import ParNotSet


@contextmanager
def _temp_mod_numpy_print_ops(np_print_threshold=None):
    np_mod = _sys.modules.get('numpy')
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

    def filler_calc(key_width):
        return ''.join(['\n', ' ' * (key_width + 2 + align_padding_width)])

    keys = sorted(data.keys(), key=_key or str) if sort else list(data.keys())

    key_widths = [len(repr(key)) for key in keys]
    # temporarily modify numpy print threshold
    with _temp_mod_numpy_print_ops(np_print_threshold=np_print_threshold):
        if align_values:
            width = max(key_widths) if key_widths else 0
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
    _base_dict_methods = ['__init__', '__setattr__', '__setitem__', '__getitem__', '__delitem__', 'update', 'pop',
                          'popitem', 'setdefault', 'clear', 'fromkeys']
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

        # extract cached method references
        if issubclass(_base_dict, dict):
            if not hasattr(kls, '_base_dict'):
                setattr(kls, '_base_dict', _base_dict)
                for method, _based_dict_method in cls._base_dict_method_map.items():
                    setattr(kls, method, getattr(_base_dict, _based_dict_method))

        _internal_name_set = set(kls._internal_names_set) if hasattr(kls, '_internal_names_set') else set()
        _dir_kls_set = set(dir(kls))

        if _dir_kls_set.difference(_internal_name_set):
            kls._internal_names_set = _internal_name_set.union(_dir_kls_set)
        return kls


class StructDictMixin(metaclass=StructDictMeta):
    __slots__ = ()
    __internal_names = []
    _internal_names_set = set(__internal_names)

    @classmethod
    def _constructor(cls, *args, **kwargs):
        obj = cls.__new__(cls)
        obj._base_dict_init(*args, **kwargs)
        return obj

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self._base_dict_init(*args, **kwargs)

    @property
    def base_dict(self):
        """Return the base_dict of the struct_dict."""
        return self._base_dict


    @abstractmethod
    def setdefault(self, key, default=None):
        if key in self:
            return self._base_dict_setdefault(key, default)
        else:
            ret_val = self._base_dict_setdefault(key, default)
            if key in self._internal_names_set:
                self._check_invalid_keys()
            return ret_val

    def __setattr__(self, key, value):
        if key in self:
            self[key] = value
            return
        try:
            attr = object.__getattribute__(self, key)
        except AttributeError:
            pass
        else:
            if isinstance(attr, (_MethodType, _BuiltinMethodType)):
                raise ValueError(
                    f"Cannot modify or add item:'{key}', it is a {self.__class__.__name__} object method.")
            else:
                return object.__setattr__(self, key, value)

        if key in self._internal_names_set:
            return object.__setattr__(self, key, value)
        else:
            self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __dir__(self):
        orig_dir = set(dir(type(self)))
        _dict_keys = set(self.__dict__.keys()) if hasattr(self, '__dict__') else set()
        _slots_keys = set(self.__slots__) if hasattr(self, '__slots__') else set()
        additions = {key for key in list(self.keys())[:100] if isinstance(key, str)}
        rv = orig_dir | _dict_keys | _slots_keys | additions
        return sorted(rv)

    @_recursive_repr()
    def __repr__(self):
        """Return string representation of struct dict.
        """
        return struct_repr(self, sort=False)

    def _sorted_repr(self):
        return struct_repr(self, sort=True)

    def __reduce__(self):
        """Support for pickle.

        The tricks played with caching references in
        :func:`StructDictMixin.__new__` confuse pickle so customize the reducer.

        """
        if hasattr(self, '_key'):
            args = (self._key, dict(self))
        else:
            args = (dict(self),)
        return (self._constructor, args)

    def copy(self):
        if hasattr(self, '_key'):
            return self._constructor(self._key, self)
        else:
            return self._constructor(self)

    __copy__ = copy

    def deepcopy(self, memo=None):
        if hasattr(self, '_key'):
            return self._constructor(self._key, _deepcopy(dict(self), memo=memo))
        else:
            return self._constructor(_deepcopy(dict(self), memo=memo))

    __deepcopy__ = deepcopy

    def get_sub_base_dict(self, keys, default=ParNotSet):
        if default is ParNotSet:
            try:
                return self._base_dict({key: self[key] for key in keys})
            except KeyError as ke:
                raise KeyError(f"Invalid key in keys: '{ke.args[0]}'")
        else:
            return self._base_dict({key: self.get(key, default) for key in keys})

    def get_sub_list(self, keys, default=ParNotSet):
        if default is ParNotSet:
            try:
                return [self[key] for key in keys]
            except KeyError as ke:
                raise KeyError(f"Invalid key in keys: '{ke.args[0]}'")
        else:
            return [self.get(key, default) for key in keys]

    def get_sub_struct(self, keys, default=ParNotSet):
        return self._constructor(self.get_sub_base_dict(keys, default=default))

    @classmethod
    def sub_struct_fromdict(cls, dict_, keys, default=ParNotSet):
        return cls.get_sub_struct(cls(dict_), keys, default=default)

    def as_base_dict(self):
        """Return a new base_dict of the struct_dict which maps item names to their values."""
        return self._base_dict(self)

    def to_reversed_map(self):
        try:
            reversed_map = self._constructor({value: key for key, value in self.items()})
        except TypeError:
            raise TypeError("Can only create reversed map of struct where all struct values are hashable")

        # noinspection PyTypeChecker
        if len(reversed_map) < len(self):
            values = list(self.values())
            duplicates = {key: item for key, item in self.items() if values.count(item) > 1}
            raise ValueError(
                f"Can only create reversed map of struct where all struct values are unique. The following keys have "
                f"duplicate values: '{set(duplicates.keys())}'.")

        return reversed_map

    @classmethod
    def combine_structs(cls, *structs):
        total_len = sum((len(struct) for struct in structs))

        combined_struct = cls._constructor(itertools.chain(*[struct.items() for struct in structs]))

        # noinspection PyTypeChecker
        if len(combined_struct) != total_len:
            all_keys = [key for struct in structs for key in struct.keys()]
            duplicates = set([key for key in all_keys if all_keys.count(key) > 1])
            raise ValueError(
                f"Can only combine structs with unique keys. The following keys are duplicated: '{duplicates}'")

        return combined_struct


class StructDict(StructDictMixin, dict):
    __slots__ = ()

    @_recursive_repr()
    def __repr__(self):
        """Return string representation of StructDict
        """
        return struct_repr(self, sort=True)


class OrderedStructDict(StructDictMixin, _OrderedDict):
    pass  # already has a __dict__ from _OrderedDict


class SortedStructDict(StructDictMixin, _SortedDict):
    # extract all internal names of _SortedDict
    __internal_names = list(_SortedDict(callable).__dict__.keys())
    _internal_names_set = StructDictMixin._internal_names_set.union(__internal_names)


# ################################################################################
# ### struct_prop_dict
# ################################################################################


def _itemsetter(item):
    def caller(self, value):
        self[item] = value

    return caller


def _itemgetter(item):
    def caller(self):
        try:
            return self[item]
        except KeyError:
            raise AttributeError

    return caller


def _itemdeleter(item):
    def caller(self):
        del self[item]

    return caller


def add_item_accessor_property(obj, item):
    try:
        setattr(obj, item, _property(fget=_itemgetter(item),
                                     fset=_itemsetter(item),
                                     doc=f"Alias for self['{item}']"))
    except TypeError:
        pass


class StructPropDictMeta(StructDictMeta):
    def __new__(cls, name, bases, _dict, **kwargs):
        kls = super(StructPropDictMeta, cls).__new__(cls, name, bases, _dict, **kwargs)

        if hasattr(kls, '_field_names'):
            invalid_field_names = set(kls._field_names).intersection(kls._internal_names_set)
            if invalid_field_names:
                raise ValueError(
                    f"Cannot create StructPropDict with invalid field names, i.e. names contained in "
                    f"'_internal_names_set': '{invalid_field_names}'")
            for name in kls._field_names:
                add_item_accessor_property(kls, name)
            if '_field_names_set' in kls.__dict__:
                if set(kls._field_names).difference(kls._field_names_set):
                    raise ValueError("All items in _field_names must be in _field_names_set.")
            else:
                kls._field_names_set = frozenset(kls._field_names)
        else:
            kls._field_names = ()
            kls._field_names_set = frozenset()

        try:
            kls._default_dict = kls._base_dict.fromkeys(kls._field_names, None)
        except AttributeError:
            pass

        return kls


class StructPropDictMixin(StructDictMixin, metaclass=StructPropDictMeta):
    __slots__ = ()

    _field_names = ()
    _field_names_set = frozenset(_field_names)

    _default_dict = {}

    @abstractmethod
    def __init__(self, *args, _default=None, **kwargs):
        if _default is None:
            self._base_dict_init(self._default_dict)
        else:
            self._base_dict_init(self._base_dict.fromkeys(self._field_names, _default))
        if args and kwargs:
            self._base_dict_init(*args, **kwargs)

    def __setattr__(self, key, value):
        if key in self:
            self[key] = value
            return
        try:
            attr = object.__getattribute__(self, key)
        except AttributeError:
            pass
        else:
            if isinstance(attr, (_MethodType, _BuiltinMethodType)):
                raise ValueError(
                    f"Cannot modify or add item: '{key}', it is a {self.__class__.__name__} object method.")
            else:
                return object.__setattr__(self, key, value)

        if key in self._internal_names_set:
            return object.__setattr__(self, key, value)
        else:
            self[key] = value
            self._create_property(key)

    def __getattr__(self, key):
        try:
            retval = self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        self._create_property(key)
        return retval

    def _create_property(self, name):
        try:
            object.__getattribute__(self, name)
        except AttributeError:
            add_item_accessor_property(self.__class__, name)
        except TypeError:
            pass

    def _delete_property(self, name):
        if name not in self._field_names_set:
            try:
                delattr(self.__class__, name)
            except AttributeError:
                pass

    def _delete_redundant_properties(self):
        redundant_props = set(self.keys()).difference(self._field_names_set)
        for key in redundant_props:
            self._delete_property(key)

    def _create_required_properties(self):
        missing_props = set(self.keys()).difference(self._field_names_set)
        for key in missing_props:
            self._create_property(key)

    @abstractmethod
    def clear(self):
        self._base_dict_clear()
        self._delete_redundant_properties()

    def to_reversed_map(self):
        reversed_map = super(StructPropDictMixin, self).to_reversed_map()
        self._create_required_properties()
        return reversed_map


_struct_prop_dict_class_template = """\
from builtins import dict as BaseDict
from {structdict_module} import StructPropDictMixin

if '{base_dict}'!='dict':
    from {structdict_module} import _{base_dict} as BaseDict

class _StructPropDict(StructPropDictMixin, BaseDict):
    if BaseDict.__name__ == 'SortedDict':
        __internal_names = list(BaseDict(callable).__dict__.keys())
        _internal_names_set = StructPropDictMixin._internal_names_set.union(__internal_names)
    
    if {sorted_repr}:
        __repr__ = StructPropDictMixin._sorted_repr

class {typename}(_StructPropDict):
    __slots__ = ()
    _field_names = {field_names!r}
"""


def struct_prop_dict(typename, field_names=None, *, structdict_module='utils.structdict', base_dict=None, verbose=False,
                     rename=False, module=None):
    """Returns a new subclass of StructDict with all fields as properties."""

    # Validate the field names.  At the user's option, either generate an error
    # message or automatically replace the field name with a valid name.

    if _isclass(base_dict):
        base_dict = base_dict.__name__

    if base_dict is None:
        base_dict = 'dict'
    elif base_dict not in ('dict', 'OrderedDict', 'SortedDict'):
        raise NotImplementedError(f"base_dict: {base_dict} is not supported.")

    sorted_repr = True if base_dict in ('dict',) else False

    if isinstance(field_names, str):
        field_names = field_names.replace(',', ' ').split()
    field_names = list(map(str, field_names)) if field_names else []
    typename = str(typename)
    if rename:
        seen = set()
        for index, name in enumerate(field_names):
            if (not name.isidentifier()
                    or _iskeyword(name)
                    or name.startswith('_')
                    or name in seen):
                field_names[index] = '_%d' % index
            seen.add(name)
    for name in [typename, structdict_module] + field_names:
        if type(name) is not str:
            raise TypeError('Type names and field names and structdict_module must be strings')
        if name is not structdict_module and not name.isidentifier():
            raise ValueError('Type names and field names must be valid '
                             'identifiers: %r' % name)
        if _iskeyword(name):
            raise ValueError('Type names and field names cannot be a '
                             'keyword: %r' % name)
    seen = set()
    for name in field_names:
        if name.startswith('_') and not rename:
            raise ValueError('Field names cannot start with an underscore: '
                             '%r' % name)
        if name in seen:
            raise ValueError('Encountered duplicate field name: %r' % name)
        seen.add(name)

    # Fill-in the class template
    class_definition = _struct_prop_dict_class_template.format(
        structdict_module=structdict_module,
        base_dict=base_dict,
        typename=typename,
        field_names=tuple(field_names),
        sorted_repr=sorted_repr
    )

    # Execute the template string in a temporary namespace and support
    # tracing utilities by setting a value for frame.f_globals['__name__']
    namespace = dict(__name__='struct_prop_dict_%s' % typename)
    exec(class_definition, namespace)
    result = namespace[typename]
    result._source = class_definition
    if verbose:
        print(result._source)

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named structdict is created.  Bypass this step in environments where
    # _sys._getframe is not defined (Jython for example) or _sys._getframe is not
    # defined for arguments greater than 0 (IronPython), or where the user has
    # specified a particular module.
    if module is None:
        try:
            module = _sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass
    if module is not None:
        result.__module__ = module

    return result


def struct_prop_ordereddict(typename, field_names=None, *, structdict_module='utils.structdict', verbose=False,
                            rename=False, module=None):
    return struct_prop_dict(typename, field_names=field_names, structdict_module=structdict_module, verbose=verbose,
                            rename=rename, module=module, base_dict=_OrderedDict)


if __name__ == '__main__':
    st = StructDict(a=1, b=2)
    sst = SortedStructDict(b=1, a=2)
