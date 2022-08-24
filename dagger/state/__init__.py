"""The dagger state is the component responsible for keeping track of
running graphs and ops.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Mapping, Optional
from collections import namedtuple

from ..graph import InstanceId, GraphInstance
from ..logger import LogEntry, LogEventType
from ..util import loads_json


class InstanceState(Enum):
    """Enumeration describing the state of a running instance."""

    RECEIVED = "RECEIVED"  # The instance has been created, but no op has executed
    STARTED  = "STARTED"   # Some ops have started
    COMPLETE = "COMPLETE"  # All output ops have completed
    ERROR    = "ERROR"     # One of the ops threw an error that was not recoverable

    STALE    = "STALE"     # The instance was started with an older
                           # version of the code which no worker can
                           # run.

    TIMEOUT = "TIMEOUT"    # The instance took longer than we expected
                           # and timed out

    @staticmethod
    def advances_state(old: "InstanceState", new: "InstanceState") -> bool:
        """Return true if the state update from old to new should actually be applied."""
        if new == InstanceState.RECEIVED and old != InstanceState.RECEIVED:
           return True
        elif new == InstanceState.STARTED and \
             old not in (InstanceState.RECEIVED, InstanceState.STARTED):
            return True
        elif new == InstanceState.COMPLETE and old != InstanceState.COMPLETE:
            return True
        elif new in (InstanceState.ERROR, InstanceState.STALE, InstanceState.TIMEOUT) and \
             old in (InstanceState.RECEIVED, InstanceState.STARTED):
            return True


InstanceDetail = namedtuple("InstanceDetail", ("state", "metadata",))


class State(ABC):
    """Keeps track of state for all instances and ops.

    This reflects the current state, whereas the log reflects the
    changes in state.

    From a log, you can always reconstruct the state.
    """

    __slots__ = ('_context',)

    def set_context(self, context: "dagger.executor.ExecutionContext"):
        self._context = context

    def log_event(self, event: LogEntry):
        """Log the given event to the state and make any changes."""
        if event.event_type == LogEventType.GRAPH_INSTANCE_STARTED:
            if event.instance is not None:
                self.set_instance_state(event.instance, InstanceState.RECEIVED, event.data)
        elif event.event_type == LogEventType.GRAPH_INSTANCE_COMPLETED:
            if event.instance is not None:
                if event.data['disposition'] == 'success':
                    self.set_instance_state(event.instance, InstanceState.COMPLETE, {})
                elif event.data['disposition'] == 'failure':
                    self.set_instance_state(event.instance, InstanceState.ERROR, {})
                elif event.data['disposition'] == 'timeout':
                    self.set_instance_state(event.instance, InstanceState.TIMEOUT, {})
        elif event.event_type == LogEventType.GRAPH_INSTANCE_SUSPENDED:
            if event.instance is not None:
                if event.data['reason'] == 'stale':
                    self.set_instance_state(event.instance, InstanceState.STALE, {})
        elif event.event_type == LogEventType.OP_STARTED:
            if event.instance is not None:
                self.set_instance_state(event.instance, InstanceState.STARTED, {})
        elif event.event_type == LogEventType.OP_RESULT:
            if event.instance is not None and event.op is not None and \
               'result' in event.data and 'opid' in event.data and \
               'port' in event.data:
                # TODO dynamic outputs can yield more than one
                # OP_RESULT. Make sure the result is the complete one.

                self.save_op_result(event.instance, event.data['opid'], event.data['port'],
                                    event.data['result'],
                                    order=event.data.get('order', None))
                self.step_instance_state(event.instance, event.data['opid'],
                                         complete=False, portname=event.data['port'])
        elif event.event_type == LogEventType.OP_COMPLETED:
            if event.instance is not None and event.op is not None and \
               'opid' in event.data:
                self.step_instance_state(event.instance, event.data['opid'], complete=True)

    def _merge_metadata(self, old: Mapping[str, Any], new: Mapping[str, Any]):
        """Merge event metadata."""
        merged = dict(new)

        if 'instance' not in new and 'instance' in old:
            merged['instance'] = old['instance']

        return merged

    @abstractmethod
    def wait_for_update(self, instance: InstanceId):
        """Wait for an update to the state of the given instance."""
        pass

    def get_instance(self, instance: InstanceId) -> GraphInstance:
        """Get an instance from the store.

        If the instance records are not found, or they don't contain
        the instance, raises KeyError.
        """
        st = self.get_instance_state(instance)
        if st is None or 'instance' not in st.metadata:
            raise KeyError(instance)

        return loads_json(st.metadata['instance'], GraphInstance, repo=self._context.repo)

    @abstractmethod
    def step_instance_state(self, instance: InstanceId,
                            opid: int, complete: bool, portname: Optional[str]=None):
        """Step the instance state, marking the given op as successful or not."""
        pass

    @abstractmethod
    def set_instance_state(self, instance: InstanceId,
                           state: InstanceState,
                           metadata: Mapping[str, Any]):
        """Set the state for the given instance."""
        pass

    @abstractmethod
    def get_instance_state(self, instance: InstanceId) -> Optional[InstanceDetail]:
        """Lookup state and metadata for the given instance."""
        pass

    @abstractmethod
    def save_op_result(self, instance: InstanceId, opid: int, portname: str, result: Any,
                       order: Optional[int]=None):
        """Store a result in the state."""
        pass

    @abstractmethod
    def load_op_result(self, instance: InstanceId, opid: int, portname: str) -> Any:
        """Load an operation result."""
        pass
