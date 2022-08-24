"""Utilities"""

from typing import Optional, Any, Mapping, Iterable
from typing.io import TextIO
from abc import ABC, abstractmethod

import datetime
import json
import enum

from .repo import Repository


class JSONMissingKeys(Exception):
    """Error called to indicate missing keys on an object."""

    __slots__ = ('class_', 'keys',)

    def __init__(self, cls: type, keys: Iterable[str]):
        """Initialize the exception with the given class and keys."""
        self.class_ = cls
        self.keys = keys

    def __str__(self):
        """Return the human-readable version of the exception."""
        return f"<JSONMissingKeys class={self.class_} keys={self.keys}>"


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for dagger types."""

    def __init__(self, repo: Optional[Repository] = None, **kwargs):
        """Initialize the encoder with a repository."""
        super().__init__(**kwargs)
        self.repo = repo

    @classmethod
    def _encode(self, o: Any):
        """Encode the given object."""

        if hasattr(o, '__json__'):
            o = o.__json__(self)
        elif isinstance(o, datetime.datetime):
            o = o.timestamp()
        elif hasattr(o, '_fields'):
            # Named tuple
            o = { f: getattr(o, f) for f in o._fields }
        elif isinstance(o, enum.Enum):
            o = o.value

        return o

    def default(self, o: Any):
        """Encode the given object the superclass doesn't know how to handle."""
        orig = o
        o = self._encode(orig)
        if o is orig:
            super().default(o)
        return o

    def iterencode(self, o: Any, **kwargs):
        """Iteratively encode o according to our rules."""
        o = self._encode(o)
        yield from super().iterencode(o, **kwargs)

    @classmethod
    def type(self, ty: Any):
        """Encode the given type (either a builtin type or a type hint)."""
        return self.pointer(ty)  # TODO

    @classmethod
    def typed(self, d: Any):
        """Encode the given object with a type annotation.

        This allows the decoder to automatically deserialize the type.
        """
        return { "__type__": self.pointer(type(d)),
                 "__data__": d }

    @classmethod
    def pointer(self, *args, **kwargs):
        """Encode the given object as a pointer to a module/name pair, like pickle."""
        from importlib import import_module
        if len(kwargs) == 0:
            if len(args) != 1:
                raise TypeError("pointer() expects an object with __module__ and __name__")

            o = args[0]
            if not hasattr(o, '__module__') or \
               not hasattr(o, '__name__'):
                raise TypeError("pointer() expects an object with __module__ and __name__")

            module_name = o.__module__
            name = o.__name__
            original = o
        else:
            if len(args) > 0:
                raise TypeError("pointer() takes one argument or keywords module and name")

            if 'module' in kwargs:
                module_name = kwargs['module']
                del kwargs['module']
            else:
                raise TypeError("pointer() expects a 'module' argument")

            if 'name' in kwargs:
                name = kwargs['name']
                del kwargs['name']
            else:
                raise TypeError("pointer() expects a 'name' argument")

            if 'original' in kwargs:
                original = kwargs['original']
                del kwargs['original']
            else:
                original = None

            if len(kwargs) > 0:
                raise TypeError("pointer() got extra arguments {}".format(", ".join(kwargs.keys())))

        if not isinstance(module_name, str) or not isinstance(name, str):
            raise TypeError("'module' and 'name' parameters must be strings")

        module = import_module(module_name)
        if not hasattr(module, name):
            raise KeyError(name)

        o = getattr(module, name)
        if original is not None:
            if o is not original:
                raise ValueError(f"Cannot serialize {original} by pointer because the retrieved value is not the same")

        return {"__pointer__": True,
                "module": module_name,
                "name": name}

class JSONDecoder(json.JSONDecoder):
    """Custom JSON decoder for dagger types."""

    def __init__(self, repo: Optional[Repository] = None, **kwargs):
        """Initialize the decoder with a repository."""
        super().__init__(**kwargs)
        self.repo = repo

    def _custom_decode(self, d: Mapping[str, Any]):
        """Custom decode any object."""

        next = d
        if "__pointer__" in d and d["__pointer__"]:
            from importlib import import_module
            if 'module' not in d or 'name' not in d:
                raise json.JSONDecodeError("__pointer__ object not valid", d, 0)
            if not isinstance(d['module'], str) or \
               not isinstance(d['name'], str):
                raise json.JSONDecodeError("Invalid __pointer__ object", d, 0)

            m = import_module(d['module'])
            if not hasattr(m, d['name']):
                raise json.JSONDecodeError(f"Could not import {d['name']} from {d['module']}", d, 0)

            next =  getattr(m, d['name'])
        elif "__type__" in d and "__data__" in d:
            ty = d['__type__']
            data = d['__data__']
            next = self.as_(data, ty)

        return next

    def as_(self, data: Any, t: type):
        """Return the deserialized data as the given type."""
        if isinstance(data, t):
            return data
        elif hasattr(t, '__readjson__'):
            return t.__readjson__(data, self)
        elif hasattr(t, '__fromjson__'):
            n = t.__new__(t)  # type: ignore
            n.__fromjson__(data, self)
            return n
        elif hasattr(t, '_fields'):
            # Namedtuple
            kwargs = {f: data[f] for f in t._fields if f in data}
            return t(**kwargs)
        elif t is datetime.datetime:
            return datetime.datetime.fromtimestamp(data)
        else:
            return t(data)


class JSONSerializable(ABC):
    """Base class for objects that can be serialized into JSON."""

    @abstractmethod
    def __json__(self, enc: JSONEncoder) -> Any:
        """Return the internal state of the object as a JSON-serializable structure."""
        pass


class JSONDeserializable(ABC):
    """Base class for objects that can be deserialized from JSON."""

    @abstractmethod
    def __fromjson__(self, d: Any, dec: JSONDecoder):
        """Load the state of the object from the state given."""
        pass


def load_json(fp, ty = None, repo = None):
    dec = JSONDecoder(repo=repo)
    data = json.load(fp, object_hook=dec._custom_decode)
    if ty is None:
        return data
    else:
        return dec.as_(data, ty)

def loads_json(s, ty = None, repo = None):
    dec = JSONDecoder(repo=repo)
    data = json.loads(s, object_hook=dec._custom_decode)
    if ty is None:
        return data
    else:
        return dec.as_(data, ty)

def dump_json(d: Any, fp: TextIO):
    json.dump(d, fp, cls=JSONEncoder)

def dumps_json(d: Any):
    return json.dumps(d, cls=JSONEncoder)
