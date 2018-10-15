import collections


class TransformedDict(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.stor2 = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.stor2[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        print('here')
        self.stor2[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.stor2[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.stor2)

    def __len__(self):
        return len(self.stor2)

    def __keytransform__(self, key):
        return key

class MyTransformedDict(TransformedDict):

    def __keytransform__(self, key):
        return key.lower()

s = MyTransformedDict()