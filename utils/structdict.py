from sortedcontainers import SortedDict
import re
import types

# with meta class CleanSetAttrMeta, __settattr__ and __getattribute__ are inherited from super class for
# __new__ and __init__
class CleanSetAttrMeta(type):
    def __call__(cls, *args, **kwargs):
        _real_setattr_ = cls.__setattr__
        _real_getattr_ = cls.__getattr__
        cls.__setattr__ = object.__setattr__
        cls.__getattr__ = object.__getattribute__
        self = super(CleanSetAttrMeta, cls).__call__(*args, **kwargs)
        cls.__setattr__ = _real_setattr_
        cls.__getattr__ = _real_getattr_
        return self


# with meta class CleanSetAttrMeta, __settattr__ and __getattribute__ are inherited from super class for
# __new__ and __init__
class StructDict(dict, metaclass=CleanSetAttrMeta):
    '''A dict with dot access and autocompletion.

    Taken from:
    https://gist.github.com/golobor/397b5099d42da476a4e6
    '''

    def __init__(self, *args, **kwargs):
        super(StructDict, self).__init__(*args, **kwargs)
        _sdict = dict
        self._sdict_getitem = types.MethodType(_sdict.__getitem__, self)
        self._sdict_setitem = types.MethodType(_sdict.__setitem__, self)
        self._sdict_getattribute = types.MethodType(_sdict.__getattribute__, self)
        self._sdict_setattr = types.MethodType(_sdict.__setattr__, self)

    def __setattr__(self, key, value):
        try:
            self._sdict_getattribute(key)
            self._sdict_setattr(key, value)
        except AttributeError:
            self._sdict_setitem(key, value)

    def __getattr__(self, key):
        # only called when an attribute is NOT found in the instance's dictionary
        try:
            return self._sdict_getitem(key)
        except KeyError:
            raise AttributeError("Attribute with key: '{}', does not exist".format(key))

    def __dir__(self):
        orig_dir = set(dir(type(self)))
        __dict__keys = set(self.__dict__.keys())
        additions = set(self.keys())
        rv = orig_dir | __dict__keys | additions
        return sorted(rv)

    def __repr__(self):
        type_name = type(self).__name__
        return ''.join([type_name, '(', self._sdict.__repr__(), ')'])


class SortedStructDict(SortedDict, metaclass=CleanSetAttrMeta):

    def __init__(self, *args, **kwargs):
        super(SortedStructDict, self).__init__(*args, **kwargs)
        _sdict = SortedDict
        self._sdict_getitem =  types.MethodType(_sdict.__getitem__, self)
        self._sdict_setitem =  types.MethodType(_sdict.__setitem__, self)
        self._sdict_getattribute =  types.MethodType(_sdict.__getattribute__, self)
        self._sdict_setattr =  types.MethodType(_sdict.__setattr__, self)

    def __setattr__(self, key, value):
        try:
            self._sdict_getattribute(key)
            self._sdict_setattr(key, value)
        except AttributeError:
            self._sdict_setitem(key, value)

    def __getattr__(self, key):
        # only called when an attribute is NOT found in the instance's dictionary
        try:
            return self._sdict_getitem(key)
        except KeyError:
            raise AttributeError("Attribute with key: '{}', does not exist".format(key))



    def __dir__(self):
        orig_dir_set = set(dir(type(self)))
        __sdict_keys_set = set(self.__dict__.keys())
        additions_set = set(self.keys())
        rv = orig_dir_set | __sdict_keys_set | additions_set
        return sorted(rv)


class StructDictAliased(StructDict, metaclass=CleanSetAttrMeta):
    _metadata = ("_striped_keys")
    _reg_num = re.compile('[^0-9]')

    def __init__(self, *args, **kwargs):
        super(StructDictAliased, self).__init__(*args, **kwargs)
        self._striped_keys = self._get_striped_key_map()
        non_unique_aliased_keys = set(self.keys()).difference(self._striped_keys.values())
        if len(non_unique_aliased_keys) > 0:
            raise ValueError('Cannot add items with duplicate aliases: {}'.format(non_unique_aliased_keys))

    def _get_striped_key_map(self):
        return {self._strip_key(key): key for key in self.keys()}

    def _strip_key(self, key):
        split_key = key.split('_', 1)
        return ''.join([split_key[0], self._reg_num.sub('', ''.join(split_key[1:]))])

    def __setattr__(self, key, value):
        try:
            self._sdict_getattribute(key)
            self._sdict_setattr(key, value)
        except AttributeError:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        striped_key = self._strip_key(key)
        if striped_key in self._striped_keys.keys():
            self._sdict_setitem(self._striped_keys[striped_key], value)
        else:
            self._striped_keys[striped_key] = key
            self._sdict_setitem(key, value)

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError("Attribute with alias/key: '{}', does not exist".format(key))

    def __getitem__(self, key):
        try:
            return self._sdict_getitem(key)
        except KeyError:
            striped_key = self._strip_key(key)
            try:
                return self._sdict_getitem(self._striped_keys[striped_key])
            except KeyError:
                raise KeyError("Key with alias: '{}', does not exist".format(key))

    def get(self, key, default = None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

if __name__ == '__main__':
    pass
    # std = StructDict(a=1, b=2)
    # stda = StructDictAliased(a=1, b=2)
    # # aliased_struct_sdict = StructDictAliased(a=1,b=2,a13=3)
    # sstd = SortedStructDict(a=1)
    # sstda = SortedStructDictAliased(a=1)
