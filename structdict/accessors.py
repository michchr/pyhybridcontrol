class AttributeAccessor:
    __slots__ = ('name', 'doc')

    def __init__(self, name, doc=None):
        if not isinstance(name, str):
            raise TypeError(f"'name' must be of type {str.__name__!r} or a subclass, not: {type(name).__name__!r}")
        else:
            self.name = name

        if doc is None:
            self.doc = f"""Attribute accessor for variable {name!r}."""
        elif not isinstance(doc, str):
            raise TypeError(f"'doc' must be of type {str.__name__!r} or a subclass.")
        else:
            self.doc = doc

    @property
    def __doc__(self):
        return self.doc

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            try:
                return instance.__dict__[self.name]
            except (AttributeError, KeyError):
                raise AttributeError(
                    f"{instance.__class__.__name__!r} object has no attribute {self.name!r} in instance "
                    f"'_dict__'")

    def __set__(self, instance, value):
        try:
            instance.__dict__[self.name] = value
        except AttributeError:
            raise AttributeError(
                f"{instance.__class__.__name__!r} object has no attribute {self.name!r} in instance "
                f"'_dict__'")

    def __delete__(self, instance):
        try:
            del instance.__dict__[self.name]
        except (AttributeError, KeyError):
            raise AttributeError(
                f"{instance.__class__.__name__!r} object has no attribute {self.name!r} in instance "
                f"'_dict__'")

    def __set_name__(self, owner, name):
        if name != self.name:
            raise ValueError(
                f"name:{name!r} of {self.__class__.__name__!r} object in class {owner.__name__!r} does not"
                f" match self.name:{self.name!r}")


class ItemAccessorMixin:
    __slots__ = ()

    def __setattr__(self, key, value):
        if hasattr(type(self), key):
            return object.__setattr__(self, key, value)
        elif isinstance(self, dict):
            self[key] = value
        else:
            return object.__setattr__(self, key, value)

    def __getattribute__(self, key):
        if hasattr(type(self), key):
            return object.__getattribute__(self, key)
        elif isinstance(self, dict):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {key!r}")
        else:
            return object.__getattribute__(self, key)


_is_using_c = False
try:
    import structdict._accessors as _accessors
    AttributeAccessor = _accessors.AttributeAccessor
    ItemAccessorMixin = _accessors.ItemAccessorMixin
    _is_using_c = True
except (ImportError, AttributeError):
    raise ImportWarning("Could not import accessors c extension.")


def is_using_c():
    return _is_using_c
