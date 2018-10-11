from sortedcontainers import SortedDict


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
        additions = {key for key in list(self.keys())[:100] if isinstance(key, str)}
        rv = orig_dir | __dict__keys | additions
        return sorted(rv)


class StructDictAliasedMixin(StructDictMixin):

    @staticmethod
    def key_aliaser_func_default(key):
        try:
            split_key = key.split('_', 1)
            no_alpha = [i for i in "".join(split_key[1:]) if i.isnumeric()]
            return ''.join(split_key[:1] + no_alpha)
        except AttributeError:
            return key

    def __init__(self, *args, key_aliaser_func=None, **kwargs):
        self._is_init_ = False
        _sdict = super(StructDictAliasedMixin, self)
        _sdict.__init__(*args, **kwargs)

        self.key_aliaser_func = key_aliaser_func or self.key_aliaser_func_default
        self._sdict_setitem = _sdict.__setitem__
        self._sdict_getitem = _sdict.__getitem__
        self._sdict_contains = _sdict.__contains__
        self._sdict_update = _sdict.update
        self._sdict_clear = _sdict.clear

        self._striped_key_map = self._get_striped_key_map()
        try:
            self._verify_stripped_keys_unique()
        except ValueError as ve:
            self._sdict_clear()
            raise ve

        self._is_init_ = True

    def _get_striped_key_map(self):
        return {self._strip_key(key): key for key in self.keys()}

    def _verify_stripped_keys_unique(self):
        non_unique_aliased_keys = set(self.keys()).difference(self._striped_key_map.values())
        if len(non_unique_aliased_keys) > 0:
            raise ValueError('Cannot add items with duplicate aliases: {}'.format(non_unique_aliased_keys))

    def _strip_key(self, key):
        return self.key_aliaser_func(key)

    def __setitem__(self, key, value):
        striped_key = self._strip_key(key)
        try:
            key_actual = self._striped_key_map[striped_key]
            self._sdict_setitem(key_actual, value)
        except KeyError:
            self._sdict_setitem(key, value)
            self._striped_key_map[striped_key] = key

    def __getitem__(self, key):
        try:
            return self._sdict_getitem(key)
        except KeyError:
            striped_key = self._strip_key(key)
            try:
                return self._sdict_getitem(self._striped_key_map[striped_key])
            except KeyError:
                raise KeyError("Key with alias: '{}', does not exist".format(key))

    def update(self, *args, **kwargs):
        if not self:
            self._sdict_update(*args, **kwargs)
            self._striped_key_map = self._get_striped_key_map()
            try:
                self._verify_stripped_keys_unique()
            except ValueError as ve:
                self._sdict_clear()
                raise ve
            return

        if not kwargs and len(args) == 1 and isinstance(args[0], dict):
            pairs = args[0]
        else:
            pairs = dict(*args, **kwargs)

        if (10 * len(pairs)) > len(self):
            self._sdict_update(pairs)
            self._striped_key_map = self._get_striped_key_map()
            try:
                self._verify_stripped_keys_unique()
            except ValueError as ve:
                self._sdict_clear()
                raise ve
            return
        else:
            for key in pairs:
                self._setitem(key, pairs[key])

    def __contains__(self, key):
        if self._sdict_contains(key):
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
    def __init__(self, *args, **kwargs):
        self._is_init_ = False
        super(StructDict, self).__init__(*args, **kwargs)
        self._is_init_ = True

    def __repr__(self):
        type_name = type(self).__name__
        return ''.join([type_name, '(', super().__repr__(), ')'])


class SortedStructDict(StructDictMixin, SortedDict):

    def __init__(self, *args, **kwargs):
        self._is_init_ = False
        super(SortedStructDict, self).__init__(*args, **kwargs)
        self._is_init_ = True


class StructDictAliased(StructDictAliasedMixin, dict):
    def __init__(self, *args, key_aliaser_func=None, **kwargs):
        self._is_init_ = False
        super(StructDictAliased, self).__init__(*args, key_aliaser_func=key_aliaser_func, **kwargs)
        self._is_init_ = True

    def __repr__(self):
        type_name = type(self).__name__
        return ''.join([type_name, '(', super().__repr__(), ')'])


class SortedStructDictAliased(StructDictAliasedMixin, SortedDict):
    def __init__(self, *args, key_aliaser_func=None, **kwargs):
        self._is_init_ = False
        super(SortedStructDictAliased, self).__init__(*args, key_aliaser_func=key_aliaser_func, **kwargs)
        self._is_init_ = True


if __name__ == '__main__':
    st = StructDict(a=1, b=2)
    sta = StructDictAliased(a=1, b=2)
    sst = SortedStructDict(b=1, a=2)
    ssta = SortedStructDictAliased(b=1, a_1a1=2)

    A = SortedStructDictAliased()
    A._sdict_setitem('c', 1)
