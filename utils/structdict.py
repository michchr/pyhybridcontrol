from sortedcontainers import SortedDict


class StructDictMixin:
    _internal_names = []
    _internal_names_set = set(_internal_names)

    def _init_std_attributes(self):
        pass
    # noinspection PyUnresolvedReferences
    def __new__(cls, *args, **kwargs):
        self = super(StructDictMixin, cls).__new__(cls, *args, **kwargs)
        _struct_dict_settattr = cls.__setattr__
        cls.__setattr__ = object.__setattr__
        #Initialize underlying dictionary and store cached access methods
        _base_dict = super(StructDictMixin, self)
        _base_dict.__init__()
        self._base_dict_setitem = _base_dict.__setitem__
        self._base_dict_getitem = _base_dict.__getitem__
        self._base_dict_contains = _base_dict.__contains__
        self._base_dict_update = _base_dict.update
        self._base_dict_clear = _base_dict.clear
        self._base_dict_init = _base_dict.__init__
        self._key = None
        self._init_std_attributes()
        cls.__setattr__ = _struct_dict_settattr

        return self

    def __setattr__(self, key, value):
        try:
            self.__getattribute__(key)
            return object.__setattr__(self, key, value)
        except AttributeError:
            pass

        if key in self._internal_names_set:
            object.__setattr__(self, key, value)
        else:
            self.__setitem__(key, value)

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

        self._base_dict_init(*args, **kwargs)
        object.__setattr__(self, '_key_aliaser_func', key_aliaser_func or self.key_aliaser_func_default)
        object.__setattr__(self, '_striped_key_map', self._get_striped_key_map())

        try:
            self._verify_stripped_keys_unique()
        except ValueError as ve:
            self._base_dict_clear()
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
            self._base_dict_setitem(key_actual, value)
        except KeyError:
            self._base_dict_setitem(key, value)
            self._striped_key_map[striped_key] = key

    def __getitem__(self, key):
        try:
            return self._base_dict_getitem(key)
        except KeyError:
            striped_key = self._strip_key(key)
            try:
                return self._base_dict_getitem(self._striped_key_map[striped_key])
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
            self._base_dict_update(*args, **kwargs)
            self._striped_key_map = self._get_striped_key_map()
            try:
                self._verify_stripped_keys_unique()
            except ValueError as ve:
                self._base_dict_clear()
                raise ve
            return

        if not kwargs and len(args) == 1 and isinstance(args[0], dict):
            pairs = args[0]
        else:
            pairs = dict(*args, **kwargs)

        # noinspection PyTypeChecker
        # Len inherited from associated dict class
        if (10 * len(pairs)) > len(self):
            self._base_dict_update(pairs)
            self._striped_key_map = self._get_striped_key_map()
            try:
                self._verify_stripped_keys_unique()
            except ValueError as ve:
                self._base_dict_clear()
                raise ve
            return
        else:
            for key in pairs:
                self.__setitem__(key, pairs[key])

    def __contains__(self, key):
        if self._base_dict_contains(key):
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
    def __repr__(self):
        type_name = type(self).__name__
        return ''.join([type_name, '(', super(StructDictMixin, self).__repr__(), ')'])


class SortedStructDict(StructDictMixin, SortedDict):
    pass

class StructDictAliased(StructDictAliasedMixin, dict):
    def __repr__(self):
        type_name = type(self).__name__
        return ''.join([type_name, '(', super(StructDictAliased, self).__repr__(), ')'])

class SortedStructDictAliased(StructDictAliasedMixin, SortedDict):
    pass

if __name__ == '__main__':
    st = StructDict(a=1, b=2)
    sta = StructDictAliased(a=1, b=2)
    sst = SortedStructDict(b=1, a=2)
    ssta = SortedStructDictAliased(b=1, a_1a1=2)

    A = SortedStructDictAliased()
    A._base_dict_setitem('c', 1)
