import sys as _sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict as _OrderedDict
from copy import deepcopy as _deepcopy, copy as _copy
from keyword import iskeyword as _iskeyword
from reprlib import recursive_repr as _recursive_repr
import inspect
import itertools

from sortedcontainers import SortedDict as _SortedDict
from structdict.accessors import AttributeAccessor, ItemAccessorMixin
from utils.func_utils import ParNotSet
import warnings as _warnings

# Ensure numpy deprecation warning is caught for array comparison
_warnings.filterwarnings('error', category=DeprecationWarning,
                         message="The truth value of an empty array is ambiguous.")

_default_structdict_np_printoptions = dict(threshold=36,
                                           precision=6,
                                           linewidth=120)


def struct_repr(data, type_name=None, sort=False, np_printoptions=None, align_values=True, align_padding_width=0,
                value_format_str=None, repr_format_str=None):
    if not isinstance(data, dict):
        return repr(data)

    if np_printoptions is None:
        np_printoptions = _default_structdict_np_printoptions

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

    def format_items():
        if align_values:
            width = max(key_widths) if key_widths else 0
            fill = filler_calc(width)
            items = ',\n'.join(item_format(key=key, value=data[key], width=width).replace('\n', fill) for key in keys)
        else:
            width = 0
            items = ',\n'.join(
                item_format(key=key, value=data[key], width=width).replace('\n', filler_calc(key_width)) for
                key, key_width in zip(keys, key_widths))

        return items

    np_mod = _sys.modules.get('numpy')
    # temporarily modify default numpy print options if numpy module exists
    if np_mod:
        with np_mod.printoptions(**np_printoptions):
            items = format_items()
    else:
        items = format_items()

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

        def _get_base_dict(_mro):
            for ind, _klass in enumerate(reversed(_mro)):
                if type(_klass) == StructDictMeta:
                    for _klass in _mro[-ind:]:
                        if issubclass(_klass, dict):
                            return _klass
                    break
            return None

        # this gives the base_dict as follows:
        _base_dict = _get_base_dict(mro)

        # extract cached method references
        if _base_dict and not hasattr(kls, '_base_dict'):
            setattr(kls, '_base_dict', _base_dict)
            for method, _based_dict_method in cls._base_dict_method_map.items():
                setattr(kls, method, getattr(_base_dict, _based_dict_method))

        all_slots = set(itertools.chain.from_iterable(klass.__dict__.get("__slots__", ()) for klass in mro))
        all_slots.discard('__dict__')
        kls._all_slots = tuple(all_slots)

        _internal_name_set = (
            all_slots.union(kls._internal_names_set) if hasattr(kls, '_internal_names_set') else all_slots)
        _internal_names = getattr(kls, "".join(["_", kls.__name__, "__internal_names"]), None)
        if _internal_names:
            for name in _internal_names:
                if getattr(kls, name, None) is None:
                    p = AttributeAccessor(name=name)
                    setattr(kls, name, p)
            _internal_name_set.update(_internal_names)

        _other_internal_names = (
            set(kls.__bases__[0]._internal_names_set) if hasattr(kls.__bases__[0], '_internal_names_set') else set())
        if _internal_name_set.difference(_other_internal_names):
            kls._internal_names_set = _internal_name_set.union(_other_internal_names)

        return kls


class StructDictMixin(ItemAccessorMixin, metaclass=StructDictMeta):
    __slots__ = ()
    _all_slots = ()  # _all_slots is set by metaclass

    __internal_names = ()
    # _internal_names_set is updated with all subclass internal names by metaclass
    _internal_names_set = set(__internal_names)


    def _get_slot_dict(self):
        # _all_slots is set by metaclass
        if self._all_slots:
            return {name: getattr(self, name, ParNotSet) for name in self._all_slots}
        else:
            return None

    @classmethod
    def _constructor(cls, items=None, copy_items=False, deepcopy_items=False,
                     instance_dict=None, instance_slots=None,
                     copy_instance_attr=False, deepcopy_instance_attr=False,
                     items_override=None,
                     inst_dict_attr_override=None,
                     inst_slot_attr_override=None,
                     memo=None):

        obj = cls.__new__(cls)

        if instance_dict is not None:
            if deepcopy_instance_attr:
                instance_dict = _deepcopy(instance_dict, memo=memo)
            elif copy_instance_attr:
                instance_dict = {name: _copy(item) for name, item in instance_dict.items()}

            obj.__dict__.update(instance_dict)

        if instance_slots is not None:
            if deepcopy_instance_attr:
                instance_slots = _deepcopy(instance_slots, memo=memo)
            elif copy_instance_attr:
                instance_slots = {name: _copy(item) for name, item in instance_slots.items()}

            for name, value in instance_slots.items():
                if value is not ParNotSet:
                    setattr(obj, name, value)

        if items is not None:
            if items.__class__ is not cls._base_dict:
                items = cls._base_dict(items)
            if deepcopy_items:
                items = _deepcopy(items, memo=memo)
            elif copy_items:
                if cls._base_dict is dict:
                    items = {name: _copy(item) for name, item in items.items()}
                else:
                    items = [(name, _copy(item)) for name, item in items.items()]
            obj._base_dict_update(items)

        if inst_dict_attr_override is not None:
            obj.__dict__.update(inst_dict_attr_override)

        if inst_slot_attr_override is not None:
            for name, value in inst_slot_attr_override.items():
                setattr(obj, name, value)

        if items_override is not None:
            obj._base_dict_update(items_override)

        return obj

    def _constructor_from_self(self, items=None, copy_items=False, deepcopy_items=False,
                               instance_dict=None, instance_slots=None,
                               copy_instance_attr=False, deepcopy_instance_attr=False,
                               items_override=None,
                               inst_dict_attr_override=None,
                               inst_slot_attr_override=None,
                               memo=None):

        items = items if items is not None else self.as_base_dict()
        instance_dict = instance_dict if instance_dict is not None else getattr(self, '__dict__', None)
        instance_slots = instance_slots if instance_slots is not None else self._get_slot_dict()

        return self._constructor(items=items, copy_items=copy_items, deepcopy_items=deepcopy_items,
                                 instance_dict=instance_dict, instance_slots=instance_slots,
                                 copy_instance_attr=copy_instance_attr, deepcopy_instance_attr=deepcopy_instance_attr,
                                 items_override=items_override, inst_dict_attr_override=inst_dict_attr_override,
                                 inst_slot_attr_override=inst_slot_attr_override, memo=memo)

    @property
    def base_dict(self):
        """Return the base_dict of the struct_dict."""
        return self._base_dict

    def __dir__(self):
        orig_dir = set(super(StructDictMixin, self).__dir__())
        try:
            _dict_keys = set(self.__dict__).union(['__dict__'])
        except (AttributeError, TypeError):
            _dict_keys = set()

        try:
            _slots_keys = set(self._all_slots).union(['__slots__'])
        except (AttributeError, TypeError):
            _slots_keys = set()

        additions = {key for key in list(self.keys())[:100] if isinstance(key, str)}
        rv = orig_dir | _dict_keys | _slots_keys | additions | self._internal_names_set
        return sorted(rv)

    @_recursive_repr()
    def __repr__(self):
        """Return string representation of struct dict.
        """
        return struct_repr(self, sort=False)

    def _sorted_repr(self):
        return struct_repr(self, sort=True)

    def __eq__(self, other):
        try:
            return self._base_dict.__eq__(self, other)
        except (ValueError, DeprecationWarning) as er:
            np_mod = _sys.modules.get('numpy')
            if np_mod:
                for key, value in other.items():
                    if not isinstance(value, np_mod.ndarray):
                        if self[key] != value:
                            return False
                    elif not np_mod.array_equal(self[key], value):
                        return False
                return True
            else:
                raise er

    def __reduce__(self):
        """Support for pickle.

        The tricks played with caching references in
        :func:`StructDictMixin.__new__` confuse pickle so customize the reducer.

        """
        instance_dict = getattr(self, '__dict__', None)
        instance_slots = self._get_slot_dict()
        args = (dict(self), False, instance_dict, instance_slots)

        return (self._constructor, args)

    def copy(self):
        return self._constructor_from_self()

    __copy__ = copy

    def deepcopy(self, memo=None):
        return self._constructor_from_self(deepcopy_items=True, deepcopy_instance_attr=True, memo=memo)

    __deepcopy__ = deepcopy

    def get_sub_base_dict(self, keys, default=ParNotSet):
        if default is ParNotSet:
            try:
                if self._base_dict is dict:
                    return self._base_dict({key: self[key] for key in keys})
                else:
                    return self._base_dict([(key, self[key]) for key in keys])
            except KeyError as ke:
                raise KeyError(f"Invalid key in keys: '{ke.args[0]}'")
        else:
            if self._base_dict is dict:
                return self._base_dict({key: self.get(key, default) for key in keys})
            else:
                return self._base_dict([(key, self.get(key, default)) for key in keys])

    def get_sub_list(self, keys, default=ParNotSet):
        if default is ParNotSet:
            try:
                return [self[key] for key in keys]
            except KeyError as ke:
                raise KeyError(f"Invalid key in keys: '{ke.args[0]}'")
        else:
            return [self.get(key, default) for key in keys]

    def get_sub_struct(self, keys, default=ParNotSet):
        return self._constructor_from_self(self.get_sub_base_dict(keys, default=default))

    @classmethod
    def sub_struct_fromdict(cls, dict_, keys, default=ParNotSet):
        return cls.get_sub_struct(cls(dict_), keys, default=default)

    @classmethod
    def fromkeys_withfunc(cls, keys, func=None):
        if callable(func):
            if cls._base_dict is dict:
                return cls._constructor({key: func(key) for key in keys})
            else:
                return cls._constructor([(key, func(key)) for key in keys])
        else:
            return cls.fromkeys(keys, func)

    def as_base_dict(self):
        """Return a new base_dict of the struct_dict which maps item names to their values."""
        return self._base_dict(self)

    def to_reversed_map(self):
        try:
            if self._base_dict is dict:
                reversed_map = self._constructor_from_self({value: key for key, value in self.items()})
            else:
                reversed_map = self._constructor_from_self([(value, key) for key, value in self.items()])
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


class StructPropDictMeta(StructDictMeta):
    def __new__(cls, name, bases, _dict, **kwargs):

        kls = super(StructPropDictMeta, cls).__new__(cls, name, bases, _dict, **kwargs)

        base_kls = kls.__bases__[0]
        base_mro = base_kls.mro()
        base_field_names = tuple(
            itertools.chain.from_iterable(klass.__dict__.get("_field_names_set", ()) for klass in base_mro))

        dir_base_kls = set(dir(base_kls)).difference(base_field_names)

        if kls.__dict__.get('_field_names'):
            invalid_field_names = set(kls._field_names).intersection(kls._internal_names_set | dir_base_kls)
            if invalid_field_names:
                raise ValueError(
                    f"Cannot create StructPropDict with invalid field names, i.e. names contained in "
                    f"'_internal_names_set or base class __dict__'s : '{invalid_field_names}'")

            if '_field_names_set' in kls.__dict__:
                if set(kls._field_names).difference(kls._field_names_set):
                    raise ValueError("All items in _field_names must be in _field_names_set.")

            kls._field_names = tuple(kls._field_names) + tuple(base_field_names)
            kls._field_names_set = frozenset(kls._field_names)
        else:
            kls._field_names = base_field_names
            kls._field_names_set = frozenset(base_field_names)

        return kls


class StructPropDictMixin(StructDictMixin, metaclass=StructPropDictMeta):
    __internal_names = ()
    __slots__ = ()

    _field_names = ()
    _field_names_set = frozenset(_field_names)
    _additional_created_properties = set()

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self._base_dict_init(*args, **kwargs)

class StructPropFixedDictMixin(StructPropDictMixin):
    __internal_names = ()
    __slots__ = ()
    _field_names = ()
    _field_names_set = frozenset(_field_names)

    @abstractmethod
    def pop(self, key):
        if key in self._field_names_set:
            try:
                ret_val = self[key]
            except KeyError:
                raise KeyError(key) from None
            else:
                self[key] = None
                return ret_val
        else:
            return self._base_dict_pop(key)

    @abstractmethod
    def popitem(self):
        try:
            key, value = self._base_dict_popitem()
        except KeyError:
            raise KeyError(f"popitem(): {self.__class__} object is empty")
        else:
            self[key] = None
        return (key, value)

    @abstractmethod
    def _reset(self):
        self.__init__()

    @abstractmethod
    def clear(self):
        super(StructPropFixedDictMixin, self).clear()
        self._reset()

    @abstractmethod
    def __delitem__(self, key):
        if key in self._field_names_set:
            try:
                self[key] = None
            except KeyError:
                raise KeyError(key) from None
        else:
            return self._base_dict_delitem(key)


_struct_prop_dict_class_template = """\
from builtins import dict as BaseDict
from {structdict_module} import {mixin_type} as StructMixin

if '{base_dict}'!='dict':
    from {structdict_module} import _{base_dict} as BaseDict

class _StructPropDict(StructMixin, BaseDict):
    if BaseDict.__name__ == 'SortedDict':
        __internal_names = list(BaseDict(callable).__dict__.keys())
        _internal_names_set = StructPropDictMixin._internal_names_set.union(__internal_names)
   
class {typename}(_StructPropDict):
    '{typename}(*args, {kwargs_map} **kwargs)'
    __slots__ = ()
    _field_names = {field_names!r}
    
    def __init__(self, *args, {kwargs_map} **kwargs):
        'Initialize new instance of {typename}(*args, {kwargs_map} **kwargs)'
        self._base_dict_init({kwargs_eq_map})
        self._base_dict_update(*args, **kwargs)

    if {sorted_repr}:
        __repr__ = StructMixin._sorted_repr
        
"""


def struct_prop_dict(typename, field_names=None, default=None, fixed=False, *, structdict_module=__name__,
                     base_dict=None, sorted_repr=None, verbose=False, rename=False, module=None):
    """Returns a new subclass of StructDict with all fields as properties."""

    # Validate the field names.  At the user's option, either generate an error
    # message or automatically replace the field name with a valid name.

    if fixed:
        mixin_type = StructPropFixedDictMixin.__name__
    else:
        mixin_type = StructDictMixin.__name__

    if inspect.isclass(base_dict):
        base_dict = base_dict.__name__

    if base_dict is None:
        base_dict = 'dict'
    elif base_dict not in ('dict', 'OrderedDict', 'SortedDict'):
        raise NotImplementedError(f"base_dict: {base_dict} is not supported.")

    if sorted_repr is None:
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
                field_names[index] = f"_{index}"
            seen.add(name)
    for name in [typename, structdict_module] + field_names:
        if type(name) is not str:
            raise TypeError('Type names, field names and structdict_module must be strings')
        if name is not structdict_module and not name.isidentifier():
            raise ValueError(f"Type names and field names must be valid identifiers: {name!r}")
        if _iskeyword(name):
            raise ValueError(f"Type names and field names cannot be a keyword: {name!r}")
    seen = set()
    for name in field_names:
        if name.startswith('_') and not rename:
            raise ValueError(f"Field names cannot start with an underscore: {name!r}")
        if name in seen:
            raise ValueError(f"Encountered duplicate field name: {name!r}")
        seen.add(name)

    default_val = "None" if default is None else 'default_val'

    # Fill-in the class template
    class_definition = _struct_prop_dict_class_template.format(
        structdict_module=structdict_module,
        mixin_type=mixin_type,
        base_dict=base_dict,
        typename=typename,
        field_names=tuple(field_names),
        kwargs_map=(", ".join([f"{field_name}={default_val}" for field_name in field_names]).replace("'", "")) + (
            "," if field_names else ""),
        kwargs_eq_map=(", ".join([f"{field_name}={field_name}" for field_name in field_names]).replace("'", "")) + (
            "," if field_names else ""),
        sorted_repr=sorted_repr
    )

    # Execute the template string in a temporary namespace and support
    # tracing utilities by setting a value for frame.f_globals['__name__']
    namespace = dict(__name__=f"struct_prop_dict_{typename}")
    namespace.update(default_val=default)
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


def struct_prop_fixed_dict(typename, field_names=None, default=None, fixed=True, *,
                           structdict_module=__name__, base_dict=None, sorted_repr=False,
                           verbose=False, rename=False, module=None):
    return struct_prop_dict(typename, field_names=field_names, default=default, fixed=fixed,
                            structdict_module=structdict_module, base_dict=base_dict, sorted_repr=sorted_repr,
                            verbose=verbose,
                            rename=rename, module=module)


def struct_prop_ordereddict(typename, field_names=None, default=None, fixed=False, *,
                            structdict_module=__name__, base_dict=_OrderedDict, sorted_repr=False,
                            verbose=False, rename=False, module=None):
    return struct_prop_dict(typename, field_names=field_names, default=default, fixed=fixed,
                            structdict_module=structdict_module, base_dict=base_dict, sorted_repr=sorted_repr,
                            verbose=verbose,
                            rename=rename, module=module)


if __name__ == '__main__':
    st = StructDict(a=1, b=2)
    ost = OrderedStructDict(b=1, a=2)
    sst = SortedStructDict(b=1, a=2)

from collections import namedtuple