from . import Executor, AsyncInstanceResult
from ..graph import GraphInstance, InstanceId, InputLocation
from ..logger import LogEntry, LogEventType
from ..util import dumps_json, loads_json, JSONEncoder
from ..storage import Storage, ProvenanceInfo
from .queued import OpResult

import queue
import sys
import logging

from typing import Optional, Iterable, Mapping, Sequence
from collections import defaultdict
from multiprocessing import pool, Pool, cpu_count, Queue
from threading import Thread, Lock, Condition


log = logging.getLogger(__name__)


def _run_op_in_worker(
    instance_id: InstanceId,
    nodeid: int,
    nodedef: str,
    storagedefs: Sequence[str],
    result_storage: Mapping[str, int],
    arglocs: Iterable[InputLocation],
    kwarglocs: Mapping[str, InputLocation],
):
    """Run operations in a worker."""
    global __dagger_queue__
    global __dagger_repo__

    results: Queue = __dagger_queue__

    # Load all storages
    storages: list[Storage] = []
    for d in storagedefs:
        storage = loads_json(d, repo=__dagger_repo__)
        storages.append(storage)

    def load_arg(input_loc: InputLocation):
        storage = storages[input_loc.storage]
        return storage.load_result(input_loc.location)

    try:
        log.info(f"Running operation {nodeid} (instance: {instance_id})")

        op: OpDefinition = loads_json(nodedef)
        log.info(f"Running {op}")

        sig = op.get_signature()
        args = [load_arg(loc) for loc in arglocs]
        kwargs = {name: load_arg(loc) for name, loc in kwarglocs.items()}

        orders: dict[str, int] = defaultdict(lambda: 0)

        for result in op.perform(*args, **kwargs):
            storage = storages[result_storage[result.port]]

            order = None
            output_sig = sig.outputs[result.port]
            if output_sig.dynamic:
                order = orders[result.port]
                orders[result.port] += 1

            log.debug(f"Process output {result}")
            pointer = storage.store_result(
                op, output_sig,
                {
                    "instance": instance_id,
                    "op": nodeid,
                    "opname": op.item_name,
                    "port": result.port,
                    "order": order,
                    'args': arglocs,
                    'kwargs': kwarglocs,
                },
                result.data,
            )

            # Write result to queue
            results.put(
                OpResult(
                    instance_id=instance_id,
                    node_id=nodeid,
                    port=result.port,
                    success=pointer,
                    failure=None,
                    complete=False,
                    order=order
                )
            )

        results.put(
            OpResult(
                instance_id=instance_id,
                node_id=nodeid,
                success=None,
                port=None,
                failure=None,
                complete=True,
            )
        )

    except:
        import traceback

        e = sys.exc_info()
        try:
            results.put(
                OpResult(
                    instance_id=instance_id,
                    node_id=nodeid,
                    complete=True,
                    port=None,
                    success=None,
                    failure=(e[1], traceback.format_exc()),
                )
            )
        except:
            print("Exception, while handling op exception")
            traceback.print_exc()

            print("Old exception was:")
            traceback.print_exc(*e)


class MultiprocessExecutor(Executor):
    """Execute a graph in a multiprocessing context."""

    __slots__ = (
        "_pool",
        "_worker_count",
        "_context",
        "_manager",
        "_lock",
        "_result_thread",
        "_active",
        "_active_changed",
        "_quitting",
    )

    _pool: Optional[pool.Pool]
    _worker_count: int

    def __init__(self, context, *, worker_count=None):
        """Initialize a multiprocessing pool to execute graphs."""
        if worker_count is None:
            worker_count = cpu_count()

        self._worker_count = worker_count
        self._pool = None
        self._lock = self._active_changed = None
        self._result_thread = None
        self._active = None
        self._quitting = True

        self._context = context

        super().__init__(context)

    def _initialize_worker(self, queue, repo):
        """Initialize the worker's global namespace with our manager."""
        global __dagger_queue__
        global __dagger_repo__
        __dagger_queue__ = queue
        __dagger_repo__ = repo

    def _result_thread_worker(self, result_queue: Queue):
        """Read results from the queue and step states as appropriate."""
        while True:
            # Wait for results from the queue
            try:
                result: OpResult = result_queue.get(True, 5)
            except queue.Empty:
                with self._lock:
                    if self._quitting:
                        return
                    else:
                        continue

            log.debug(f"Processing result {result}")

            with self._lock:
                key = (result.instance_id, result.node_id)
                if key not in self._active:
                    print(f"{key} not a running operation.")

                try:
                    instance: GraphInstance = self.context.state.get_instance(
                        result.instance_id
                    )
                    instance.bind_executor(self)
                except KeyError:
                    print(f"Instance {result.instance_id} not found, skipping")
                    continue  # TODO log

                # Step the graph
                if result.failure is not None:
                    e, tb = result.failure
                    print(f"{key} fails\n", tb)  # TODO what to do when a graph fails
                    print(result.failure[0])
                    print(result.failure[1])

                # Log the event
                elif result.port is not None:
                    self.context.log_event(
                        LogEntry(
                            event_type=LogEventType.OP_RESULT,
                            graph=instance.graph.item_name,
                            instance=result.instance_id,
                            op=instance[result.node_id].item_name,
                            data={
                                "executor": "MultiprocessExecutor",
                                "opid": result.node_id,
                                "port": result.port,
                                "order": result.order,
                                "result": result.success
                            },
                        )
                    )

                if result.complete:
                    # Log the operation as having succeeded
                    self._active.remove(key)

                    log.info(f"{key} succeeds")
                    self.context.log_event(
                        LogEntry(event_type=LogEventType.OP_COMPLETED,
                                 graph=instance.graph.item_name,
                                 instance=result.instance_id,
                                 op=instance[result.node_id].item_name,
                                 data={
                                     'executor': 'MultiprocessExecutor',
                                     'opid': result.node_id,
                                 }))

                # Check if any new nodes became ready and schedule them
                try:
                    next_instance: GraphInstance = self.context.state.get_instance(result.instance_id)
                    next_instance.bind_executor(self)
                except KeyError:
                    continue

            if next_instance.all_outputs_complete():
                self.context.log_event(
                    LogEntry(event_type=LogEventType.GRAPH_INSTANCE_COMPLETED,
                             graph=instance.graph.item_name,
                             instance=result.instance_id,
                             data={
                                 'executor': 'MultiprocessExecutor',
                                 'disposition': 'success'
                             })
                )
            else:
                newly_ready = {nodeid for nodeid, _ in next_instance.ready()}
                log.debug("These nodes are ready %s", newly_ready)
                for nodeid, _ in instance.ready():
                    newly_ready.discard(nodeid)
                log.debug("These nodes became ready %s", newly_ready)

                self._launch_all(instance, newly_ready)

    def __enter__(self):
        """Enable execution within the context."""

        queue = Queue()
        self._pool = Pool(
            self._worker_count, initializer=self._initialize_worker, initargs=(queue, self.context.repo,)
        )

        self._quitting = False

        self._lock = Lock()
        self._active_changed = Condition()
        self._active = set()

        self._result_thread = Thread(target=self._result_thread_worker, args=(queue,))
        self._result_thread.start()

    def __exit__(self, exc_type, exc_value, traceback):
        """Delete the pool."""
        with self._lock:
            self._quitting = True

        self._pool.close()
        self._pool.join()

        # Signal the result thread to end once all results are received
        self._result_thread.join()

        self._pool = self._lock = self._result_thread = self._active_changed = None

    def start(self, instance: GraphInstance):
        """Launch the given graph instance in the pool."""
        if self._pool is None:
            raise RuntimeError(
                "You must use contexts to work with a MultiprocessExecutor"
            )

        instance_id = GraphInstance.allocate_id()

        instance.bind(instance_id, self)

        self.context.log_event(
            LogEntry(
                event_type=LogEventType.GRAPH_INSTANCE_STARTED,
                graph=instance.graph.item_name,
                instance=instance_id,
                data={
                    "executor": "MultiprocessExecutor",
                    "instance": dumps_json(instance),
                },
            )
        )

        # Run each portion of the graph
        ready = {nodeid for nodeid, _ in instance.ready()}
        self._launch_all(instance, ready)

        return AsyncInstanceResult(instance_id, executor=self)

    def _launch_all(self, instance: GraphInstance, ready: set[int]):
        """Launch all ready portions of the graph."""
        assert self._pool is not None

        for nodeid in ready:
            node = instance[nodeid]

            with self._lock:
                self._active.add((instance.id, nodeid))

            # Prepare arguments
            storages, args, kwargs, output_stores = instance._prepare_arguments(nodeid)

            a = self._pool.apply_async(
                _run_op_in_worker,
                (
                    instance.id,
                    nodeid,
                    dumps_json(node),
                    [dumps_json(JSONEncoder.typed(s)) for s in storages],
                    output_stores,
                    args,
                    kwargs,
                ),
            )
            a.get()
