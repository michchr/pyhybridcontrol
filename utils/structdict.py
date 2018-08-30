from sortedcontainers import SortedDict
import re

class StructDictMixin:
    _is_init_ = False
    def __setattr__(self, key, value):
        if self._is_init_:
            try:
                self.__getattribute__(key)
                object.__setattr__(self, key, value)
            except AttributeError:
                self.__setitem__(key, value)
        else:
            object.__setattr__(self, key, value)

    def __getattr__(self, key):
        # only called when an attribute is NOT found in the instance's dictionary
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError("Attribute with key: '{}', does not exist".format(key))

    def __dir__(self):
        orig_dir = set(dir(type(self)))
        __dict__keys = set(self.__dict__.keys())
        additions = {key for key in self.keys()[:100] if isinstance(key, str)}
        rv = orig_dir | __dict__keys | additions
        return sorted(rv)


class StructDictAliasedMixin(StructDictMixin):
    _metadata = ("_striped_keys")
    _reg_num = re.compile('[^0-9]')

    _is_init_ = False
    def __init__(self, *args, **kwargs):
        super(StructDictAliasedMixin, self).__init__(*args, **kwargs)
        self._is_init_ = False
        self._striped_keys = self._get_striped_key_map()
        non_unique_aliased_keys = set(self.keys()).difference(self._striped_keys.values())
        if len(non_unique_aliased_keys) > 0:
            raise ValueError('Cannot add items with duplicate aliases: {}'.format(non_unique_aliased_keys))

        _dict = super(StructDictAliasedMixin, self)
        self._sdict_setitem = _dict.__setitem__
        self._sdict_getitem = _dict.__getitem__
        self._sdict_contains = _dict.__contains__

        self._is_init_= True

    def _get_striped_key_map(self):
        return {self._strip_key(key): key for key in self.keys()}

    def _strip_key(self, key):
        try:
            split_key = key.split('_', 1)
            return ''.join([split_key[0], self._reg_num.sub('', ''.join(split_key[1:]))])
        except AttributeError:
            return key

    def __setitem__(self, key, value):
        striped_key = self._strip_key(key)
        if striped_key in self._striped_keys.keys():
            self._sdict_setitem(self._striped_keys[striped_key], value)
        else:
            self._sdict_setitem(key, value)
            self._striped_keys[striped_key] = key

    def __getitem__(self, key):
        try:
            return self._sdict_getitem(key)
        except KeyError:
            striped_key = self._strip_key(key)
            try:
                return self._sdict_getitem(self._striped_keys[striped_key])
            except KeyError:
                raise KeyError("Key with alias: '{}', does not exist".format(key))

    def __contains__(self, key):
        if self._sdict_contains(key):
            return True
        else:
            striped_key = self._strip_key(key)
            if striped_key in self._striped_keys:
                return True
            else:
                return False

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

class StructDict(StructDictMixin, dict):
    def __init__(self, *args, **kwargs):
        super(StructDict, self).__init__(*args, **kwargs)
        self._is_init_ = True

    def __repr__(self):
        type_name = type(self).__name__
        return ''.join([type_name, '(', super().__repr__(), ')'])

class SortedStructDict(StructDictMixin, SortedDict):

    _is_init_ = False
    def __init__(self, *args, **kwargs):
        super(SortedStructDict, self).__init__(*args, **kwargs)
        self._is_init_ = True

class StructDictAliased(StructDictAliasedMixin, dict):
    def __init__(self, *args, **kwargs):
        super(StructDictAliased, self).__init__(*args, **kwargs)
        self._is_init_ = True

    def __repr__(self):
        type_name = type(self).__name__
        return ''.join([type_name, '(', super().__repr__(), ')'])

class SortedStructDictAliased(StructDictAliasedMixin, SortedDict):

    _is_init_ = False
    def __init__(self, *args, **kwargs):
        super(SortedStructDictAliased, self).__init__(*args, **kwargs)
        self._is_init_ = True





if __name__ == '__main__':
    st = StructDict(a=1, b=2)
    sta = StructDictAliased(a=1, b=2)
    sst = SortedStructDict(b=1, a=2)
    ssta = SortedStructDictAliased(b=1, a_1a1=2)

    A = SortedStructDictAliased()
    A._sdict_setitem('a', 1)