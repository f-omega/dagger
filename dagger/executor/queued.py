"""Provides a base class for any executor that operates based off of a shared queue."""
from collections import namedtuple, defaultdict
from typing import Sequence, Mapping, Any, Iterable, Optional
from abc import abstractmethod
import sys

from . import Executor
from ..storage import Storage
from ..op import OpDefinition
from ..graph import GraphInstance, InstanceId, InputLocation
from ..logger import LogEntry, LogEventType
from ..util import JSONEncoder, dumps_json, loads_json


OpResult = namedtuple(
    "OpResult",
    (
        "instance_id",
        "node_id",
        # Optional past hre
        "port",
        "complete",
        "success",
        "failure",
        "order"
    ),
    defaults=(
        None,
        None,
        None,
        None,
        None,
    ),
)


class OpRequest(namedtuple("OpRequest",
                           ("instance_id",
                            "nodeid",
                            "nodedef",
                            "storagedefs",
                            "result_storage",
                            "arglocs",
                            "kwarglocs",
                            # Optional
                            "metadata"),
                           defaults=({},))):
    instance_id: InstanceId
    nodeid: int
    nodedef: str
    storagedefs: Sequence[str]
    result_storage: Mapping[str, int]
    arglocs: Sequence[InputLocation]
    kwarglocs: Mapping[str, InputLocation]
    metadata: Mapping[str, Any]


class QueuedExecutorMixin(Executor):
    """Mix-in class providing support for shared-queue executors."""

    @abstractmethod
    def _send_result(self, request: OpRequest, result: OpResult):
        """Send a result back to the result queue."""
        pass

    @abstractmethod
    def _launch_all(self, reqs: Iterable[OpRequest]):
        """Launch the given requests."""
        pass

    def _make_request(self, instance: GraphInstance, nodeid: int):
        """Make an OpRequest for the given node in the instance."""
        node = instance[nodeid]
        storages, args, kwargs, output_stores = instance._prepare_arguments(nodeid)

        return OpRequest(instance_id=instance.id,
                         nodeid=nodeid,
                         nodedef=dumps_json(node),
                         storagedefs=[dumps_json(JSONEncoder.typed(s)) for s in storages],
                         result_storage=output_stores,
                         arglocs=args,
                         kwarglocs=kwargs)

    def _load_json(self, o: str, ty: Optional[type] = None) -> Any:
        """Load JSON object from the given string (sets context as appropriate)."""
        return loads_json(o, ty, repo=self.context.repo)

    def _load_arg(self, storages: Sequence[Storage],
                  argloc: InputLocation) -> Any:
        storage = storages[argloc.storage]
        return storage.load_result(argloc.location)

    def run_op_in_worker(self, request: OpRequest):
        """Process a particular operation."""
        storages = [self._load_json(d) for d in request.storagedefs]

        op: OpDefinition = self._load_json(request.nodedef)
        sig = op.get_signature()

        args = [self._load_arg(storages, loc) for loc in request.arglocs]
        kwargs = {name: self._load_arg(storages, loc) for name, loc in request.kwarglocs.items()}

        orders: dict[str, int] = defaultdict(lambda: 0)
        try:
            for result in op.perform(*args, **kwargs):
                result_storage = storages[request.result_storage[result.port]]

                order = None
                output_sig = sig.outputs[result.port]
                if output_sig.dynamic:
                    order = orders[result.port]
                    orders[result.port] += 1

                pointer = result_storage.store_result(
                    op, output_sig,
                    {
                        "instance": request.instance_id,
                        "op": request.nodeid,
                        "opname": op.item_name,
                        "port": result.port,
                        "order": order,
                        "args": request.arglocs,
                        "kwargs": request.kwarglocs
                    },
                    result.data
                )

                # Send result to queue
                self._send_result(request,
                                  OpResult(
                                      instance_id=request.instance_id,
                                      node_id=request.nodeid,
                                      port=result.port,
                                      success=pointer,
                                      complete=False,
                                      order=order))

            self._send_result(request,
                              OpResult(
                                  instance_id=request.instance_id,
                                  node_id=request.nodeid,
                                  port=result.port,
                                  complete=True))
        except:
            import traceback
            e = sys.exc_info()

            try:
                self._send_result(request,
                                  OpResult(instance_id=request.instance_id,
                                           node_id=request.nodeid,
                                           complete=True,
                                           failure=(e[1], traceback.format_exc())))
            except:
                print("Exception, while handling op exception")
                traceback.print_exc()

                print("Old exception was:")
                traceback.print_exc(*e)

    def start(self, instance: GraphInstance):
        """Launch the given graph instance into the queue."""
        instance_id = GraphInstance.allocate_id()
        instance.bind(instance_id, self)

        self.context.log_event(
            LogEntry(
                event_type=LogEventType.GRAPH_INSTANCE_STARTED,
                graph=instance.graph.item_name,
                instance=instance_id,
                data={
                    "executor": self.__class__.__name__,
                    "instance": dumps_json(instance)
                }
            )
        )

        reqs = [self._make_request(instance, nodeid) for nodeid, _ in instance.ready()]
        self._launch_all(reqs)
