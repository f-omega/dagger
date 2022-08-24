"""Storage class to define how operations store results."""

from abc import abstractmethod
from typing import Any, Optional
from io import BytesIO

from .. import util
from ..op import ProvenanceInfo, OpDefinition, Output


class Storage(util.JSONSerializable, util.JSONDeserializable):
    """ABC for any class that implements the storage interface."""

    __slots__ = ("_context",)

    def set_context(self, context: "dagger.executor.ExecutionContext"):
        self._context = context

    @abstractmethod
    def store_result(self, op: Optional[OpDefinition], output: Optional[Output],
                     provenance: ProvenanceInfo, x: Any) -> Any:
        """Store the given result."""
        pass

    @abstractmethod
    def load_result(self, pointer: Any) -> Any:
        """Load the result using the pointer given (same as that returned from store_result)."""
        pass


class KVStorage(Storage):
    """ABC for storage classes that store by hashed op value.

    The pointer for these objects is the hashed value of the operation
    hashinfo.

    Parameters are hashed in a separate manner
    """

    __slots__ = ('keysep',)

    def __init__(self, keysep='-'):
        """Initialize the KVStorage with an optional key separator."""
        self.keysep = keysep

    @abstractmethod
    def _hash(self, b: BytesIO) -> str:
        """Hash the given bytes, producing a string representing the hash."""
        pass

    @abstractmethod
    def store_keyed_result(self, key: str, x: Any):
        """Store the given result by the given key."""
        pass

    @abstractmethod
    def load_keyed_result(self, key: str) -> Any:
        """Load the given result by key."""
        pass

    def store_result(self, op: Optional[OpDefinition], output: Optional[Output],
                     provenance: ProvenanceInfo, x: Any) -> Any:
        """Store the result by hashing the inputs."""
        b = BytesIO()
        if op is not None:
            op._mkhashinfo(provenance, b)
        else:
            # This is a parameter
            b.write(b'__param__\n')
            b.write('{}\n{}\n'.format(provenance['instance'], provenance['op']).encode('utf-8'))

        b.seek(0)
        key = self._hash(b)

        if provenance['port'] is not None:
            key = '{}{}{}'.format(key, self.keysep, provenance['port'])

        if provenance['order'] is not None:
            key = '{}{}{}'.format(key, self.keysep, provenance['order'])

        self.store_keyed_result(key, x)

        return key

    def load_result(self, pointer: Any) -> Any:
        """Load the result from the given hash string."""
        return self.load_keyed_result(pointer)

