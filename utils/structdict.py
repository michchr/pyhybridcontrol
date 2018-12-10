from types import MethodType, BuiltinMethodType
from reprlib import recursive_repr
from sortedcontainers import SortedDict
from collections import OrderedDict, namedtuple as NamedTuple
from copy import deepcopy as _deepcopy

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


def struct_repr(data, type_name=None, sort=False, np_print_threshold=20, align_values=False, align_padding_width=0,
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


class StructDictMixin:
    __internal_names = []
    _internal_names_set = set(__internal_names)

    def __init__(self, *args, **kwargs):
        super(StructDictMixin, self).__init__(*args, **kwargs)
        self._check_invalid_keys()

    def _check_invalid_keys(self, key=None):
        if self._internal_names_set:
            if key is None:
                invalid_keys = self._internal_names_set.intersection(self.keys())
            else:
                invalid_keys = key if key in self._internal_names_set else None

            if invalid_keys:
                for key in invalid_keys:
                    del self[key]
                raise ValueError(
                    "Cannot add items to struct dict with keys contained in '_internal_names_set': '{}'".format(
                        invalid_keys))
        else:
            return

    def __setitem__(self, key, value):
        # noinspection PyUnresolvedReferences
        super(StructDictMixin, self).__setitem__(key, value)
        self._check_invalid_keys(key=key)

    def update(self, *args, **kwargs):
        # noinspection PyUnresolvedReferences
        super(StructDictMixin, self).update(*args, **kwargs)
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
            self.__setitem__(key, value)

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
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

    def get_sub_dict(self, keys):
        try:
            return {key: self[key] for key in keys}
        except KeyError as ke:
            raise KeyError(f"Invalid key in keys: '{ke.args[0]}'")

    def get_sub_list(self, keys):
        try:
            return [self[key] for key in keys]
        except KeyError as ke:
            raise KeyError(f"Invalid key in keys: '{ke.args[0]}'")


    def get_sub_struct(self, keys):
        return self.__class__(self.get_sub_dict(keys))

    @classmethod
    def sub_struct_fromdict(cls, dict_, keys):
        return cls(cls.get_sub_dict(dict_, keys))

    def as_dict(self):
        return dict(self)


class StructDictAliasedMixin(StructDictMixin):
    __internal_names = ['_key_aliaser_func', '_striped_key_map']
    _internal_names_set = StructDictMixin._internal_names_set.union(__internal_names)

    @staticmethod
    def key_aliaser_func_default(key):
        try:
            split_key = key.split('_', 1)
            no_alpha = [i for i in "".join(split_key[1:]) if i.isnumeric()]
            return ''.join(split_key[:1] + no_alpha)
        except AttributeError:
            return key

    def __init__(self, *args, key_aliaser_func=None, **kwargs):
        super(StructDictAliasedMixin, self).__init__(*args, **kwargs)
        self._key_aliaser_func = key_aliaser_func or self.key_aliaser_func_default
        self._striped_key_map = self._get_striped_key_map()

        try:
            self._verify_stripped_keys_unique()
        except ValueError as ve:
            super(StructDictAliasedMixin, self).clear()
            raise ve

    def _get_striped_key_map(self):
        return {self._strip_key(key): key for key in self.keys()}

    def _verify_stripped_keys_unique(self):
        non_unique_aliased_keys = set(self.keys()).difference(self._striped_key_map.values())
        if len(non_unique_aliased_keys) > 0:
            raise ValueError('Cannot add items with duplicate aliases: {}'.format(non_unique_aliased_keys))

    def _strip_key(self, key):
        return self._key_aliaser_func(key)

    def __setitem__(self, key, value):
        striped_key = self._strip_key(key)
        try:
            key_actual = self._striped_key_map[striped_key]
            super(StructDictAliasedMixin, self).__setitem__(key_actual, value)
        except KeyError:
            super(StructDictAliasedMixin, self).__setitem__(key, value)
            self._striped_key_map[striped_key] = key

    def __getitem__(self, key):
        try:
            return super(StructDictAliasedMixin, self).__getitem__(key)
        except KeyError:
            striped_key = self._strip_key(key)
            try:
                return super(StructDictAliasedMixin, self).__getitem__(self._striped_key_map[striped_key])
            except KeyError:
                raise KeyError("Key with alias: '{}', does not exist".format(key))

    def update(self, *args, **kwargs):
        """Update struct dict aliased with items from `args` and `kwargs`.

        Overwrites existing items.

        Optional arguments `args` and `kwargs` may be a mapping, an iterable of
        pairs or keyword arguments.

        :param args: mapping or iterable of pairs
        :param kwargs: keyword arguments mapping

        Method based on sortedcontainers.SortedDict update method
        """
        if not self:
            super(StructDictAliasedMixin, self).update(*args, **kwargs)
            self._striped_key_map = self._get_striped_key_map()
            try:
                self._verify_stripped_keys_unique()
            except ValueError as ve:
                super(StructDictAliasedMixin, self).clear()
                raise ve
            return

        if not kwargs and len(args) == 1 and isinstance(args[0], dict):
            pairs = args[0]
        else:
            pairs = dict(*args, **kwargs)

        # noinspection PyTypeChecker
        # Len inherited from associated dict class
        if (10 * len(pairs)) > len(self):
            super(StructDictAliasedMixin, self).update(pairs)
            self._striped_key_map = self._get_striped_key_map()
            try:
                self._verify_stripped_keys_unique()
            except ValueError as ve:
                super(StructDictAliasedMixin, self).clear()
                raise ve
            return
        else:
            for key in pairs:
                self.__setitem__(key, pairs[key])

    def __contains__(self, key):
        if super(StructDictAliasedMixin, self).__contains__(key):
            return True
        else:
            striped_key = self._strip_key(key)
            if striped_key in self._striped_key_map:
                return True
            else:
                return False

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default


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


class StructDictAliased(StructDictAliasedMixin, dict):
    @recursive_repr()
    def __repr__(self):
        """Return string representation of StructDict
        """
        return struct_repr(self, sort=True)


class OrderedStructDictAliased(StructDictAliasedMixin, OrderedDict):
    pass


class SortedStructDictAliased(StructDictAliasedMixin, SortedDict):
    # extract all internal names of SortedDict
    __internal_names = list(SortedDict(callable).__dict__.keys())
    _internal_names_set = StructDictAliasedMixin._internal_names_set.union(__internal_names)


class FrozenStructDict(StructDict):
    def __setitem__(self, key, value):
        raise TypeError("'{0}' object does not support item assignment".format(self.__class__.__name__))

    def update(self, *args, **kwargs):
        raise TypeError("'{0}' object does not support update".format(self.__class__.__name__))


if __name__ == '__main__':
    st = StructDict(a=1, b=2)
    sta = StructDictAliased(a=1, b=2)
    sst = SortedStructDict(b=1, a=2)
    ssta = SortedStructDictAliased(b=1, a_1a1=2)
