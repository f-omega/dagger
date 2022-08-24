"""Provides a State that is usable only on the local machine, for testing."""

from . import State, InstanceState, InstanceDetail
from ..op import OpDefinition
from ..graph import InstanceId, GraphInstance
from ..util import loads_json, dumps_json

from threading import Lock, Condition
from typing import Mapping, Any, Optional


class MemoryState(State):
    """Keep track of graph instances running in the local process."""

    __slots__ = ('_lock', '_condition', '_state', '_results')

    _state: dict[InstanceId, InstanceDetail]
    _results: dict[tuple[InstanceId, int, str], Any]

    def __init__(self):
        """Initialize the in-memory state."""
        self._lock = Lock()
        self._condition = Condition(lock=self._lock)
        self._state = {}
        self._results = {}

    def step_instance_state(self, instance: InstanceId, op: int, complete: bool,
                            portname: Optional[str]=None):
        """Step the instance state atomically."""
        with self._lock:
            if instance in self._state and \
               'instance' in self._state[instance].metadata:
#                print(f"Marking ready {op} {portname}")
                i: GraphInstance = self._state[instance].metadata['instance']
                i = loads_json(i, GraphInstance, repo=self._context.repo)

                if portname is not None:
                    i.mark_complete(op, nodeport=portname)
                else:
                    defn = i[op]
                    if isinstance(defn, OpDefinition):
                        sig = defn.get_signature()
                        for o in sig.outputs:
#                            print(f"Mark ready {op} {o}")
                            i.mark_complete(op, nodeport=o)

                self._state[instance].metadata['instance'] = dumps_json(i)
                self._condition.notify_all()

    def get_instance_state(self, instance: InstanceId) -> Optional[InstanceDetail]:
        """Lookup the state for the given instance."""
#        print(f'Lookup insance state {instance}', self._state)
        with self._lock:
            return self._state.get(instance, None)

    def set_instance_state(self, instance_id: InstanceId,
                           state: InstanceState,
                           metadata: Mapping[str, Any] = {}):
        """Set the state for the given instance."""
        with self._lock:
            old_state = self._state.get(instance_id, None)

            if old_state is not None:
                metadata = self._merge_metadata(old_state.metadata, metadata)

            new_state = InstanceDetail(state=state, metadata=metadata)

            if old_state is None or \
               InstanceState.advances_state(old_state.state, state):
#                print(f'Set instance state {instance_id}')
                self._state[instance_id] = new_state

            # Notify all waiters to check if the state really changed
            self._condition.notify_all()

    def wait_for_update(self, instance: InstanceId):
        """Wait for an update to the state of the given instance."""
        with self._lock:
            if instance not in self._state:
                return  # TODO should we raise KeyError?

            state = self._state[instance]

            self._condition.wait()

            if instance not in self._state:
                return
            elif self._state[instance] is not state:
                return

            # Must have been notified for another instance. Continue

    def save_op_result(self, instance: InstanceId, opid: int, portname: str, result: Any,
                       order: Optional[int]=None):
        """Save the given operation result."""
        with self._lock:
#            print(f'save op result {instance} {opid} {portname} {result}')
            # TODO dynamic output
            self._results[(instance, opid, portname)] = result

    def load_op_result(self, instance: InstanceId, opid: int, portname: str):
        with self._lock:
            return self._results[(instance, opid, portname)]
