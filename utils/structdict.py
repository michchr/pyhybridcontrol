from reprlib import recursive_repr
from sortedcontainers import SortedDict
from collections import OrderedDict

import pprint

class StructDictMixin:
    __internal_names = []
    _internal_names_set = set(__internal_names)

    # noinspection PyUnresolvedReferences
    def _init_std_attributes(self, *args, **kwargs):
        _sbase_dict = super(StructDictMixin, self)
        self._sbase_dict_init = _sbase_dict.__init__
        self._sbase_dict_setitem = _sbase_dict.__setitem__
        self._sbase_dict_getitem = _sbase_dict.__getitem__
        self._sbase_dict_contains = _sbase_dict.__contains__
        self._sbase_dict_pop = _sbase_dict.pop
        self._sbase_dict_update = _sbase_dict.update
        self._sbase_dict_clear = _sbase_dict.clear

    # noinspection PyUnresolvedReferences
    def __new__(cls, *args, **kwargs):
        self = super(StructDictMixin, cls).__new__(cls, *args, **kwargs)
        _struct_dict_settattr = cls.__setattr__
        cls.__setattr__ = object.__setattr__
        self._init_std_attributes(*args, **kwargs)
        cls.__setattr__ = _struct_dict_settattr
        return self

    def __init__(self, *args, **kwargs):
        self._sbase_dict_init(*args, **kwargs)
        self._check_invalid_keys()

    def __reduce__(self):
        """Support for pickle.

        The tricks played with caching references in
        :func:`StructDictMixin.__new__` confuse pickle so customize the reducer.

        """
        if hasattr(self, '_key'):
            args = (self._key, list(self.items()))
        else:
            args = (list(self.items()),)
        return (self.__class__, args)

    def _check_invalid_keys(self, key=None):
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

    def __setitem__(self, key, value):
        self._sbase_dict_setitem(key, value)
        self._check_invalid_keys(key=key)

    def update(self, *args, **kwargs):
        self._sbase_dict_update(*args, **kwargs)
        self._check_invalid_keys()

    def __setattr__(self, key, value):
        try:
            self.__getattribute__(key)
        except AttributeError:
            pass
        else:
            return object.__setattr__(self, key, value)

        if key in self._internal_names_set:
            object.__setattr__(self, key, value)
        else:
            self.__setitem__(key, value)

    def __getattr__(self, key):
        # only called when an attribute is NOT found in the instance's dictionary
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            pass

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
        _key = self._key if hasattr(self, '_key') else None
        key_arg = '' if _key is None else '_key = {0!r},\n'.format(_key)
        items = pprint.pformat(dict(self), compact=True)
        type_name = type(self).__name__
        repr_format = '{0}(\n{1}{2})'.format if self else '{0}({1}{2})'.format
        return repr_format(type_name, key_arg, items)


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
    pass

class OrderedStructDict(StructDictMixin, OrderedDict):
    pass

class SortedStructDict(StructDictMixin, SortedDict):
    #extract all internal names of SortedDict
    __internal_names = list(SortedDict(callable).__dict__.keys())
    _internal_names_set = StructDictMixin._internal_names_set.union(__internal_names)


class StructDictAliased(StructDictAliasedMixin, dict):
    pass

class OrderedStructDictAliased(StructDictAliasedMixin, OrderedDict):
    pass

class SortedStructDictAliased(StructDictAliasedMixin, SortedDict):
    # extract all internal names of SortedDict
    __internal_names = list(SortedDict(callable).__dict__.keys())
    _internal_names_set = StructDictAliasedMixin._internal_names_set.union(__internal_names)

if __name__ == '__main__':
    st = StructDict(a=1, b=2)
    sta = StructDictAliased(a=1, b=2)
    sst = SortedStructDict(b=1, a=2)
    ssta = SortedStructDictAliased(b=1, a_1a1=2)
