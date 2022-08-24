"""Storage implementation that stores everything in memory."""

from . import Storage, ProvenanceInfo
from ..op import Output, OpDefinition

from typing import Any, Optional


class MemoryStorage(Storage):
    """Storage class that stores everything in picklable objects.

    Not really safe to use in production."""

    def store_result(self, op: Optional[OpDefinition],
                     output: Optional[Output],
                     provenance: ProvenanceInfo, x: Any) -> Any:
        """Return the data as-is. It should be picklable."""
#        print(f"Store {provenance}, {x}")
        return x

    def load_result(self, pointer: Any) -> Any:
        """The data is stored as is, so just return."""
        return pointer

    def __json__(self, enc):
        """Serialize the storage to json."""
        return {}

    def __fromjson__(self, o, dec):
        """Do nothing."""
        pass
