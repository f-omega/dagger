"""An executor is something that can take a graph instance and run it."""

from .. import graph, logger, state, util, repo
from ..repo import get_global_repository
from ..logger import LogFilter

from abc import ABC, abstractmethod
from typing import Optional, Iterable


class ExecutionContext(object):
    __slots__ = (
        "storage",
        "state",
        "log",
        "repo",
    )

    #    storage: StorageManager
    log: logger.Logger
    storage: "Storage"
    state: "state.State"
    repo: repo.Repository

    def __init__(self, *, log, repo=None, storage=None, state=None):
        """Initialize the execution context from the given parameters."""
        self.log = log
        if repo is None:
            repo = get_global_repository()

        self.repo = repo

        if state is None:
            from ..state.memory import MemoryState
            self.state = MemoryState()
        else:
            self.state = state

        if storage is None:
            from ..storage.memory import MemoryStorage
            self.storage = MemoryStorage()
        else:
            self.storage = storage

        self.storage.set_context(self)
        self.state.set_context(self)

    def log_event(self, event: logger.LogEntry):
        """Log an event to the store, and update the state as well."""
        self.log.save_entry(event)

        # Now that the event is persisted, save it to the state.
        #
        # The log is always the authoritative source.
        self.state.log_event(event)


class Executor(ABC):
    """An executor executes graph instances in the given context."""

    __slots__ = ("context",)

    def __init__(self, context: ExecutionContext):
        """Initialize the executor's context."""
        self.context = context

    @abstractmethod
    def start(self, instance: graph.GraphInstance):
        """Starts the execution of the graph instance.

        Returns a handle to the execution.
        """
        pass


class AsyncInstanceResult(object):
    """Represents the result of an instance that's running but not computed yet."""

    __slots__ = ("instance_id", "executor", "_signature")

    def __init__(self, instance: graph.InstanceId, executor: Executor):
        """Construct a result from an ID and executor."""
        self.instance_id = instance
        self.executor = executor
        self._signature = None

    def wait(self, outputs: Optional[Iterable[str]]=None):
        """Wait until the result becomes available.

        If the run errored out, the exception is thrown.

        The default implementation gets the instance status. If
        complete, it pulls the result. Otherwise, it invokes
        wait_for_instance_state_change on the executor, in a loop.

        If outputs is not None, it should be an iterable containing
        the list of outputs to wait for.
        """
        if outputs is None:
            pending_outputs = None
        else:
            pending_outputs = set(outputs)

        while pending_outputs is None or len(pending_outputs) > 0:
            if pending_outputs is not None:
                self.executor.context.state.wait_for_update(self.instance_id)

            instance_detail: state.InstanceDetail = self.executor.context.state.get_instance_state(self.instance_id)

            if 'instance' not in instance_detail.metadata:
                self.executor.context.state.wait_for_update(self.instance_id)
                continue

            instance = util.loads_json(instance_detail.metadata['instance'], graph.GraphInstance,
                                       repo=self.executor.context.repo)

            # Check if the outputs are ready
            if pending_outputs is None:
                if 'instance' not in instance_detail.metadata:
                    raise RuntimeError("No instance present in metadata")

                pending_outputs = set(instance.graph.get_signature().outputs.keys())
                all_outputs = set(x for x in pending_outputs)

            completed_outputs = set()
            for o in pending_outputs:
                output_id, output_port = instance.graph.get_demand(o)
                if instance.is_complete(output_id, output_port):
                    completed_outputs.add(o)

            pending_outputs -= completed_outputs

        if len(all_outputs) == 1:
            default_output = list(all_outputs)[0]
            sig = instance.graph.get_signature()
            if default_output not in sig.outputs:
                raise RuntimeError("Graph output not found")

            if not sig.outputs[default_output].dynamic:
                return self._get_output(default_output, instance)

    def _get_output(self, output_name: str, i: graph.GraphInstance):
        """Get the output with the given name from the loaded instance."""
        (nodeid, portname) = i.graph.get_demand(output_name)
        storage = i.get_output_location(nodeid, portname)
        sig = i.graph.get_signature()
        output = sig.outputs[output_name]
        if output.dynamic:
            raise NotImplementedError("Can't get output of dynamic output.")
        else:
            ptr = self.executor.context.state.load_op_result(self.instance_id, nodeid, portname)
            return storage.load_result(ptr)

    def get_output(self, output_name):
        """Get the output with the given name."""
        instance_detail: state.InstanceDetail = self.executor.context.state.get_instance_state(self.instance_id)
        instance = util.loads_json(instance_detail.metadata['instance'], graph.GraphInstance)
        return self._get_output(output_name, instance)
