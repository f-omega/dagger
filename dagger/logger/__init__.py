"""Loggers log the results of a graph execution to some durable storage."""

from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from typing import Generator, Iterable, Optional, Mapping, Any
from datetime import datetime
from .. import repo, graph, util


class LogEventType(Enum):
    """Enumeration of log event types."""

    GRAPH_INSTANCE_STARTED = "GRAPH_INSTANCE_STARTED"
    GRAPH_INSTANCE_COMPLETED = "GRAPH_INSTANCE_COMPLETED"

    # The graph execution was suspended
    GRAPH_INSTANCE_SUSPENDED = "GRAPH_INSTANCE_SUSPENDED"

    OP_STARTED = "OP_STARTED"
    OP_RETRIED = "OP_RETRIED"
    OP_ERRORED = "OP_ERRORED"
    OP_GIVES_UP = "OP_GIVES_UP"
    OP_COMPLETED = "OP_COMPLETED"
    OP_RESULT = "OP_RESULT"

    COMMENT = "COMMENT"  # Random notes from the system

    SYSTEM_EVENT = "SYSTEM_EVENT"  # An important system event


class LogEntry(object):
    """Log Entry for the dagger log."""

    __slots__ = (
        "event_type",
        "timestamp",
        "op",
        "graph",
        "instance",
        "data",
    )

    event_type: LogEventType
    timestamp: datetime
    op: Optional[repo.ItemName]
    instance: Optional["dagger.graph.InstanceId"]
    graph: Optional[repo.ItemName]
    data: Mapping[str, Any]

    def __init__(
        self,
        event_type: LogEventType,
        timestamp: datetime = None,
        op: Optional[repo.ItemName] = None,
        graph: Optional[repo.ItemName] = None,
        instance: Optional["dagger.graph.InstanceId"] = None,
        data: Mapping[str, Any] = {},
    ):
        """Initialize an event."""
        if timestamp is None:
            timestamp = datetime.now()

        self.event_type = event_type
        self.timestamp = timestamp
        self.op = op
        self.graph = graph
        self.instance = instance
        self.data = data

    def __json__(self, context=None):
        """Convert the entry into a json readable format."""
        d = {"event_type": self.event_type.value, "timestamp": self.timestamp}

        if self.op is not None:
            d["op"] = self.op

        if self.graph is not None:
            d["graph"] = self.graph

        if self.instance is not None:
            d["instance"] = self.instance

        if len(self.data) > 0:
            d["data"] = self.data

        return d

    def __fromjson__(self, data: Any, decoder: util.JSONDecoder):
        """Deserialize the log entry from json."""
        if "event_type" not in data or "timestamp" not in data:
            raise TypeError("Expected event_type and timestamp")

        self.event_type = LogEventType(data["event_type"])
        self.timestamp = decoder.as_(data["timestamp"], datetime)
        self.op = self.graph = self.instance = None

        if "op" in data:
            self.op = decoder.as_(data["op"], repo.ItemName)

        if "graph" in data:
            self.graph = decoder.as_(data["graph"], repo.ItemName)

        if "instance" in data:
            self.instance = decoder.as_(data["instance"], repo.ItemName)


class Logger(ABC):
    """Abstract class for loggers."""

    @abstractmethod
    def save_entry(self, log_entry: LogEntry):
        """Save the given log entry."""
        pass


LogFilterCapability = namedtuple(
    "LogFilterCapability",
    ("field_name", "sort_asc", "sort_desc", "fts", "exact", "range"),
)
LogFilter = namedtuple("LogFilter", ("field_name", "sort", "query",
                                     "exact", "between"),
                       defaults=(None, None, None, None,))


class LogQuerier(ABC):
    """Abstract class for logs that can be queried."""

    @abstractmethod
    def supported_filters(self) -> Generator[LogFilterCapability, None, None]:
        """Generate the list of supported filters for this log."""
        pass

    @abstractmethod
    def find(self, filters: Iterable[LogFilter]) -> Generator[LogEntry, None, None]:
        """Find all log entries matching the given filter."""
        pass
