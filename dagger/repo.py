"""A repository is a group of artifacts, ops, and jobs."""

from typing import Iterable, Optional, Dict, NewType
from abc import ABC, abstractmethod


ItemName = NewType("ItemName", str)
ItemType = NewType("ItemType", str)

class RepositoryItemName(object):
    """The name of an object in the repository."""

    item_type: ItemType
    item_name: ItemName

    def __init__(self, item_type: ItemType, item_name: ItemName):
        self.item_type = item_type
        self.item_name = item_name


class RepositoryItem(ABC):
    """Abstract base class representing an item that can be in a repository."""

    @abstractmethod
    def get_item_dependencies(self) -> Iterable['RepositoryItem']:
        """Calculate the dependencies of the item."""
        pass

    @property
    @abstractmethod
    def item_name(self) -> ItemName:
        """Get the name of the item."""
        pass

    @classmethod
    @abstractmethod
    def get_item_type(self) -> ItemType:
        """Get the type of the item."""
        pass


class Repository(object):
    """Dagger Repository class."""

    _items: Dict[str, Dict[str, RepositoryItem]] = {}

    def __init__(self, name: str,
                 contents: Optional[Iterable[RepositoryItem]] = None,
                 autodiscover=Iterable[str]):
        """Initialize a repository with the given name and contents.

        :param name: The name of the repository
        :type name: str

        :param contents: 
        """
        self._name = name
        self._items = {}

        if contents is not None:
            for c in contents:
                self.add(c)

    def __contains__(self, item: RepositoryItem):
        """Check if the given item is contained in this repository."""
        if item.get_item_type() in self._items:
            items = self._items[item.get_item_type()]
            return item.item_name in items
        else:
            return False

    def __getitem__(self, idx: tuple[str, str]):
        item_type, item_name = idx
        if item_type in self._items:
            items = self._items[item_type]
            return items[item_name]
        else:
            raise KeyError(item_type)

    def add(self, item: RepositoryItem):
        """Add the given item to the repository."""
        items = self._items.get(item.get_item_type(), {})
        self._items[item.get_item_type()] = items

        if item.item_name in items:
            raise KeyError("Item {} already exists (type: {})"
                           .format(item.item_name, item.get_item_type()))

        items[item.item_name] = item
        for d in item.get_item_dependencies():
            if d not in self:
                self.add(d)

    def iteritems(self) -> Iterable[RepositoryItem]:
        """Iterate through all items in the repository."""
        for ty in self._items:
            items = self._items[ty]

            for i in items:
                yield items[i]

_default_repository = Repository("defaults")

def get_global_repository():
    global _default_repository
    return _default_repository
