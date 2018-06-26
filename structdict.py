import re

class StructDict(dict):
    '''A dict with dot access and autocompletion.

    Taken from:
    https://gist.github.com/golobor/397b5099d42da476a4e6
    '''

    def __init__(self, *args, **kwargs):
        super(StructDict, self).__init__()
        self.update(*args, **kwargs)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, key):
        # we don't need a special call to super here because getattr is only
        # called when an attribute is NOT found in the instance's dictionary
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError("Attribute with key: '{}', does not exist".format(key))


class StructDictAliased(StructDict):
    specials = ("_striped_keys, _struct_dict")

    def __init__(self, *args, **kwargs):
        super(StructDictAliased, self).__init__(*args, **kwargs)
        self._struct_dict = super(StructDictAliased, self)
        self._striped_keys = self._get_striped_keys()

    def _get_striped_keys(self):
        return {self._strip_key(i): i for i in self.keys()}

    def _strip_key(self, key):
        split_key = key.split('_', 1)
        return ''.join([split_key[0], re.sub('[^0-9]', '', ''.join(split_key[1:]))])
        # return ''.join([key[0], re.sub('[^0-9]', '', key[1:])])

    def __setattr__(self, key, value):
        if key in StructDictAliased.specials:
            self.__dict__[key] = value
        else:
            self._striped_keys[self._strip_key(key)] = key
            self[key] = value

    def __setitem__(self, key, value):
        striped_key = self._strip_key(key)
        if striped_key in self._striped_keys.keys():
            self._struct_dict.__setitem__(self._striped_keys[striped_key], value)
        else:
            self._striped_keys[striped_key] = key
            self._struct_dict.__setitem__(key, value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("Attribute with alias: '{}', does not exist".format(key))

    def __getitem__(self, key):
        if not key in self.keys():
            striped_key = self._strip_key(key)
            try:
                return self._struct_dict.__getitem__(self._striped_keys[striped_key])
            except KeyError:
                raise KeyError("Key with alias: '{}', does not exist".format(key))
        else:
            return self._struct_dict.__getitem__(key)

    def get(self, key):
        try:
            return self[key]
        except KeyError:
            self._struct_dict.get(key)


if __name__ == '__main__':
    a = StructDictAliased(A_123asd21=1)
    a.A12_123 = 1
    a.abc = 1
    print(a._striped_keys)
    print(a.A12123)
