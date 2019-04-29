import wrapt
import functools
import itertools
import inspect
from structdict import named_struct_dict

_global_version__ = 0
VersionStruct = named_struct_dict('VersionStruct', sorted_repr=False)


def increment_version(obj):
    global _global_version__
    _global_version__ += 1
    obj._self_version__ = (id(obj), _global_version__)


def increments_version_decor(member):
    def updates_version_attr(func):
        if inspect.ismethod(func):
            func = func.__func__
        return getattr(func, '_updates_version', False)

    def set_updates_version_attr(func):
        if inspect.ismethod(func):
            func = func.__func__
        func._updates_version = True

    @wrapt.decorator
    def func_wrapper(wrapped, self, args, kwargs):
        ret_val = wrapped(*args, **kwargs)  # first process then update version
        increment_version(self)
        return ret_val

    @wrapt.decorator
    def prop_wrapper(wrapped, self, args, kwargs):
        ret_val = wrapped(*args, **kwargs)  # first process then update version
        increment_version(args[0])
        return ret_val

    if isinstance(member, property):
        fset = member.fset
        fdel = member.fdel
        if not updates_version_attr(fset):
            fset = prop_wrapper(fset)
            set_updates_version_attr(fset)

        if not updates_version_attr(fdel):
            fdel = prop_wrapper(fdel)
            set_updates_version_attr(fdel)

        return property(fget=member.fget,
                        fset=fset,
                        fdel=fdel,
                        doc=member.__doc__)

    else:
        if not updates_version_attr(member):
            wrapped = func_wrapper(member)
            set_updates_version_attr(wrapped)
            return wrapped
        else:
            return member


def get_version(obj):
    return getattr(obj, 'version', VersionStruct())


def update_stored_version(obj, from_object=None):
    current_version = get_version(from_object if from_object is not None else obj)
    obj._stored_version = current_version


def has_updated_version(obj, sub_object_names=(), current_version=None, from_object=None):
    stored_version = getattr(obj, '_stored_version', VersionStruct())
    current_version = current_version if current_version is not None else get_version(
        from_object if from_object is not None else obj)
    if not sub_object_names:
        return current_version != stored_version
    else:
        for name in sub_object_names:
            if current_version[name] != stored_version.get(name):
                return True

    return False


class VersionMixin:
    __internal_names = ('_self_version__', '_version__', '_stored_version')
    _versioned_sub_objects = ()

    def __init__(self, *args, **kwargs):
        self._init_version__()
        super(VersionMixin, self).__init__(*args, **kwargs)

    def _init_version__(self):
        if not hasattr(self, '_self_version__'):
            self._self_version__ = None
            increment_version(self)
        self._stored_version = VersionStruct()
        self._version__ = {'self_version': self._self_version__}

    @property
    def version(self):
        return VersionStruct(self._get_current_version(struct=True))

    @property
    def self_version(self):
        return self._get_current_version(struct=False)

    def increment_version(self):
        increment_version(self)

    def has_updated_version(self, sub_object_names=(), current_version=None):
        return has_updated_version(self, sub_object_names=sub_object_names, current_version=current_version)

    def update_stored_version(self):
        update_stored_version(self)

    def _get_validated_self_version(self):
        if self._self_version__[0] != id(self):
            increment_version(self)
        return self._self_version__

    def _get_current_version(self, struct=True):
        version = getattr(self, '_version__', None)
        if version is None:
            self._init_version__()
            version = self._version__
            self_version__ = self._self_version__
        else:
            version['self_version'] = self_version__ = self._get_validated_self_version()

        if not self._versioned_sub_objects:
            return version if struct else self_version__

        sub_objs = {obj_name: getattr(self, obj_name) for obj_name in self._versioned_sub_objects}
        for obj_name, obj in sub_objs.items():
            if isinstance(obj, VersionMixin):
                version[obj_name] = obj._get_sub_version()
            else:
                version[obj_name] = None

        version['self_version'] = self_version__ = (version['self_version'][0],
                                                    max([version_count[1] for version_count in
                                                         version.values() if version_count is not None]))

        return version if struct else self_version__

    def _get_sub_version(self, memo=None):
        self_version__ = getattr(self, '_self_version__', None)
        if self_version__ is None:
            self._init_version__()
            self_version__ = self._self_version__
        else:
            self_version__ = self._get_validated_self_version()

        if not self._versioned_sub_objects:
            return self_version__

        id_self = id(self)
        if memo is None:
            memo = {id_self: self_version__}
        elif id_self in memo:
            if memo[id_self][1]<=self_version__[1]:
                memo[id_self] = self_version__
                return self_version__
            else:
                self._self_version__ = self_version__ = memo[id_self]
        else:
            memo[id_self] = self_version__

        sub_objs = [getattr(self, obj_name) for obj_name in self._versioned_sub_objects if
                    isinstance(getattr(self, obj_name), VersionMixin)]

        return (self_version__[0], max(self_version__[1], *[obj._get_sub_version(memo=memo)[1] for obj in sub_objs]))


def versioned(kls=None, versioned_sub_objects=None):
    if kls is None:
        return functools.partial(versioned, versioned_sub_objects=versioned_sub_objects)

    new_kls_dict = dict(kls.__dict__)
    if versioned_sub_objects:
        v_sub = getattr(kls, '_versioned_sub_objects', None)
        if v_sub:
            v_sub = type(v_sub)(set(
                itertools.chain.from_iterable([v_sub, versioned_sub_objects])))
        else:
            v_sub = versioned_sub_objects
        new_kls_dict['_versioned_sub_objects'] = v_sub

    new_kls_dict.pop('__dict__', None)
    new_kls_dict.pop('__weakref__', None)

    bases = tuple([klass for klass in kls.__bases__ if klass is not object])
    if VersionMixin not in bases:
        bases += (VersionMixin,)

    new_kls = type(kls)(kls.__name__, bases, new_kls_dict)
    return new_kls

class VersionObject(VersionMixin):

    def __init__(self, name=None):
        super(VersionObject, self).__init__()
        self.name = name or ""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}) with version: {self.self_version}"

    def __eq__(self, other):
        if isinstance(other, (VersionObject, dict)):
            return self.version == other.version
        else:
            return self.self_version == other
