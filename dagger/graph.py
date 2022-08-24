'''A graph is a set of ops that can be run in parallel or
   sequentially, depending on their data dependencies.
'''

from .op import op, ConstantOpDefinition, OpDefinition, \
    Signature, Input, Output, ProvenanceInfo
from .repo import RepositoryItem, RepositoryItemName
from .storage import Storage
from .storage.memory import MemoryStorage
from . import util, hashes

from collections import namedtuple
from typing import NewType, Mapping, Any, Optional, Iterable, \
    Callable

import uuid
import pickle
import dis
import pytypes
import logging

InputLocation = namedtuple('InputLocation',
                           ('storage', 'location'))
log = logging.getLogger("dagger.graph")


class Param(object):
    __slots__ = ("_name", "_type")

    def __init__(self, name, ty):
        self._name = name
        self._type = ty

    def __repr__(self):
        return "<Param {}>".format(self._name)

    def __json__(self, e: util.JSONEncoder):
        """Encode the parameter in a JSON object."""
        return { "name": self._name,
                 "type": e.type(self._type) }

    def __fromjson__(self, d: Any, dec: util.JSONDecoder):
        """Read the param from the given JSON object."""
        if 'name' not in d or 'type' not in d:
            raise util.JSONMissingKeys(type(self), ("name", "type",))

        self._name = d['name']
        self._type = d['type']

class StackItem(object):
    @property
    def needs_calculation(self):
        return False

class ConstantStackItem(StackItem):
    __slots__ = ("_constant",)

    def __init__(self, c):
        self._constant = c


class GenericStackItem(StackItem):
    __slots__ = ("_opid", "_opname", "_argval", "_inputs", "_output")

    def __init__(self, opid, opname, argval, inputs, output=0):
        self._opid = opid
        self._opname = opname
        self._argval = argval
        self._inputs = inputs
        self._output = output

    @property
    def needs_calculation(self):
        return True


class OpResultStackItem(StackItem):
    __slots__ = ("_opid", "_output")

    def __init__(self, opid, output=None):
        self._opid = opid
        self._output = output

class NodeDeps(object):
    _opid: int
    _forward: set[tuple[str, int, str]]
    _backward: set[tuple[int, str, str]]
    _demands: set[str]

    __slots__ = ("_opid", "_forward", "_backward", "_demands",)

    def __init__(self, opid):
        self._opid = opid
        self._forward = set()
        self._backward = set()
        self._demands = set()

    def add_backwards(self, fromid, fromport, toport):
        self._backward.add((fromid, fromport, toport))

    def add_forwards(self, fromport, toid, toport):
        self._forward.add((fromport, toid, toport))

    def add_demand(self, port):
        self._demands.add(port)


class Graph(OpDefinition):
    __slots__ = ("_ops", "_edges",
                 "_unconnected_outputs", "_unconnected_inputs",
                 "_demands", "__wrapped__", "_signature" )

    _ops: list[OpDefinition]
    _edges: list[NodeDeps]
    _unconnected_outputs: set[tuple[int, str]]
    _unconnected_inputs: set[tuple[int, str]]
    _demands: dict[str, tuple[int, str]]
    _signature: Optional[Signature]

    def __init__(self, name: RepositoryItemName,
                 hashinfo: Callable[[OpDefinition, ProvenanceInfo], bytes] = hashes.default):
        """Initialize the graph."""
        super().__init__(name, hashinfo)

        self._ops = []
        self._edges = []

        self._signature = None

        self._unconnected_outputs = set()
        self._unconnected_inputs = set()

        self._demands = {}
        self.__wrapped__ = None

    def get_item_dependencies(self):
        """Return all ops contained here."""
        for o in self._ops:
            if isinstance(o, RepositoryItem):
                yield o

    def __getstate__(self):
        """Return a picklable representation of the graph."""
        edges = []
        for fromid, e in enumerate(self._edges):
            for fromport, toid, toport in e._forward:
                edges.append((fromid, fromport, toid, toport))

        return {"ops": self._ops, "edges": edges}

    def __setstate__(self, st):
        self.__init__()
        """Restore state from pickled representation of graph."""
        if 'ops' not in st or \
           'edges' not in st:
            raise pickle.UnpicklingError("Could not unpickle Graph from bad state")

        for op in st['ops']:
            if isinstance(op, OpDefinition):
                self.add_op(op)
            elif isinstance(op, Param):
                self.add_param(op._name, op._type)
            else:
                raise TypeError("Op must be an OpDefinition or a parameter")

        for fromid, fromport, toid, toport in st['edges']:
            self.connect(fromid, toid, fromport=fromport, toport=toport)

    def validate(self, demand_all=True):
        """Validate the graph has no unconnected inputs. Any unconnected outputs are demanded"""
        if len(self._unconnected_inputs) > 0:
            raise RuntimeError("Unconnected inputs: {}".format(self._unconnected_inputs))

        for nodeid, nodeport in self._unconnected_outputs:
            self.demand(nodeid, nodeport,
                        outname="output" if len(self._unconnected_outputs) == 1 else None)

    def write_dot(self, output):
        """Write the graph as a dot file."""
        def get_port_ix(ports, nm):
            for i, port in enumerate(ports):
                if port == nm:
                    return i
            raise KeyError(nm)
        print("digraph {", file=output)
        print("node[shape=record];", file=output)
        for i, op in enumerate(self._ops):
            if isinstance(op, Param):
                print("op{}[label=\"param {}({})\"];".format(i, op._name, str(op._type).replace("<", "&lt;").replace(">", "&gt;")), file=output)
            else:
                inputs = ""
                for j, input_ in enumerate(op.get_signature()._inputs):
                    if len(inputs) > 0:
                        inputs += "|"
                    inputs += " <i{}> {}".format(j, input_)

                outputs = ""
                for j, o in enumerate(op.get_signature()._outputs):
                    if len(outputs) > 0:
                        outputs += "|"
                    outputs += "<o{}> {}".format(j, o)

                label = f"{{ {{  {inputs} }} | {op.item_name} | {{ {outputs} }} }}"
                print("op{}[label=\"{}\"];".format(i, label), file=output)

        for fromid, edge in enumerate(self._edges):
            fromnode = self._ops[fromid]
            for fromport, toid, toport in edge._forward:
                tonode = self._ops[toid]
                if isinstance(fromnode, Param):
                    print("op{} -> op{}:i{};".format(fromid, toid,
                                                     get_port_ix(tonode.get_signature()._inputs, toport)), file=output)
                else:
                    print("op{}:o{} -> op{}:i{};".format(fromid, get_port_ix(fromnode.get_signature()._outputs, fromport),
                                                         toid, get_port_ix(tonode.get_signature()._inputs, toport)), file=output)

        print("}", file=output)

    def add_op(self, op: OpDefinition):
        """Add an OpDefinition to the graph.

        The ops inputs and outputs are considered unconnected.

        Returns the id of the operation
        """
        if not isinstance(op, OpDefinition):
            raise TypeError("add_op must be called with OpDefinition")

        if op in self._ops:
            return self._ops.index(op)

        opid = len(self._ops)
        self._ops.append(op)
        self._edges.append(NodeDeps(opid))

        for input_ in op.get_signature()._inputs:
            self._unconnected_inputs.add((opid, input_))
        for output in op.get_signature()._outputs:
            self._unconnected_outputs.add((opid, output))

        return opid

    def add_param(self, param_name, param_type):
        """Add a parameter to the graph."""
        opid = len(self._ops)
        self._ops.append(Param(param_name, param_type))
        self._edges.append(NodeDeps(opid))
        return opid

    def connect(self, fromid, toid, fromport="output", toport="input"):
        """Connect the ports of the specified nodes.

        The port being fed into must be disconnected
        """
        if fromid > len(self._ops):
            raise KeyError(fromid)

        if toid > len(self._ops):
            raise KeyError(toid)

        fromnode = self._ops[fromid]
        tonode = self._ops[toid]

        if (isinstance(fromnode, Param) and fromport != "output") or \
           (isinstance(fromnode, OpDefinition) and fromport not in fromnode.get_signature()._outputs):
            raise KeyError((fromnode, fromport))

        if toport not in tonode.get_signature()._inputs:
            raise KeyError((tonode, toport))

        if (toid, toport) not in self._unconnected_inputs:
            raise RuntimeError("{} is already connected".format(toport))

        self._edges[toid].add_backwards(fromid, fromport, toport)
        self._edges[fromid].add_forwards(fromport, toid, toport)

        self._unconnected_inputs.discard((toid, toport))
        self._unconnected_outputs.discard((fromid, fromport))

    def demand(self, fromid, fromport="output", outname=None):
        """Mark the given output as a necessary output of this graph."""
        if fromid > len(self._ops):
            raise KeyError(fromid)

        fromnode = self._ops[fromid]

        if isinstance(fromnode, Param):
            raise RuntimeError("Demanding values of parameters is not supported. Parameters must be used by ops")

        elif isinstance(fromnode, OpDefinition) and \
             fromport not in fromnode.get_signature().outputs:
            raise KeyError((fromnode, fromport))

        name = outname
        if name is None:
            name = f"{fromport}_{fromid}"

        self._unconnected_outputs.discard((fromid, fromport))
        self._demands[name] = (fromid, fromport)
        self._edges[fromid].add_demand(fromport)

    def get_demand(self, output):
        """Return the node id and port name from which an output comes."""
        if output not in self._demands:
            raise KeyError(output)

        return self._demands[output]

    @classmethod
    def from_function(self, fn):
        """Attempt to create a graph from a function."""
        bc = dis.Bytecode(fn)

        graph = self(fn.__name__)

        # Shadow stack
        stack = []
        block_stack = []
        fn_locals = [ None ] * bc.codeobj.co_nlocals

        for i in range(bc.codeobj.co_argcount):
            paramname = bc.codeobj.co_varnames[i]
            if paramname not in fn.__annotations__:
                raise TypeError("Each parameter must be annotated with a type")
            paramid = graph.add_param(paramname, fn.__annotations__[paramname])
            fn_locals[i] = OpResultStackItem(paramid, output="output")

        def make_port(arg):
            if arg.needs_calculation:
                raise NotImplementedError("Would need to make pseudo-op")
            elif isinstance(arg, OpResultStackItem):
                if arg._output is None:
                    op = graph._ops[arg._opid]
                    raise TypeError("Operation {}({}) result cannot be used directly because op has multiple outputs".format(arg._opid, op))
                else:
                    return (arg._opid, arg._output)
            elif isinstance(arg, ConstantStackItem):
                cid = graph.add_op(ConstantOpDefinition(arg._constant))
                return (cid, "output")

        generic_instructions = { 'IMPORT_NAME': (2, 2, 1),
                                 'IMPORT_FROM': (0, 1, 1),
                                 'LOAD_METHOD': (1, 1, 2) }

        log.debug("from_function(%s) starts", fn)
        for opid, instr in enumerate(bc):
            log.debug("Instruction: %s", instr)
            if instr.opname == 'LOAD_CONST':
                stack.append(ConstantStackItem(instr.argval))

            elif instr.opname == 'RETURN_VALUE':
                portid, portname = make_port(stack[-1])
                graph.demand(portid, portname)

            elif instr.opname == 'STORE_FAST':
                fn_locals[instr.arg] = stack[-1]
                stack = stack[:-1]

            elif instr.opname == 'LOAD_FAST':
                stack.append(fn_locals[instr.arg])

            elif instr.opname == 'LOAD_GLOBAL':
                try:
                    g = fn.__globals__[instr.argval]
                except KeyError:
                    raise NameError(instr.argval)
                stack.append(ConstantStackItem(g))

            elif instr.opname == 'POP_TOP':
                stack = stack[:-1]

            elif instr.opname == 'CALL_METHOD':
                arguments = stack[-instr.argval:]
                stack = stack[:-instr.argval]

                method = stack[-1]
                obj = stack[-2]
                stack = stack[:-2]

                if not isinstance(method, GenericStackItem) or \
                   not isinstance(obj, GenericStackitem) or \
                   method._opid != obj._opid or \
                   method._opname != 'LOAD_METHOD':
                    raise RuntimeError('CALL_METHOD encountered without LOAD_METHOD')

                # The question here is whether this is an op or not. Thus we have to perform a continuation
                raise NotImplementedError("Continuations...")

            elif instr.opname == 'CALL_FUNCTION':
                log.debug("CALL_FUNCTION, stack is %s", stack)
                args = []
                if instr.argval > 0:
                    args = stack[-instr.argval:]
                args.reverse()

                stack = stack[:-instr.argval]

                tgt = stack[-1]
                stack = stack[:-1]

                if isinstance(tgt, ConstantStackItem):
                    if isinstance(tgt._constant, OpDefinition):
                        # If args are constant, we're okay, or if args are the output of an op, we're okay.
                        #
                        # Otherwise, we need to make pseudo-ops

                        inputs_waiting = set(tgt._constant.get_input_order())
                        port_args = [make_port(arg) for arg in args]

                        for (_, input_) in zip(port_args, tgt._constant.get_input_order()):
                            inputs_waiting.discard(input_)

                        if len(inputs_waiting) > 0:
                            raise TypeError("Not all arguments supplied to {}".format(tgt._constant))

                        opid = graph.add_op(tgt._constant)
                        for (fromid, fromport), toport in zip(port_args, tgt._constant.get_input_order()):
                            graph.connect(fromid, opid, fromport=fromport, toport=toport)

                        outputs = tgt._constant.get_signature()._outputs
                        if len(outputs) == 0:
                            stack.append(ConstantStackItem(None)) # TODO we may want to create a temporale dependency here...
                        elif len(outputs) == 1:
                            keys = iter(outputs.keys())
                            stack.append(OpResultStackItem(opid, next(keys)))
                        else:
                            stack.append(OpResultStackItem(opid))
                    else:
                        stack.append(GenericStackItem(opid, instr.opname,
                                                      instr.argval,
                                                      [tgt] + args,
                                                      0))

                elif isinstance(fn, GenericStackItem):
                    raise NotImplementedError("Continuation...")
                else:
                    raise RuntimeError("Invalid stack value")

            elif instr.opname in generic_instructions:
                pops, reads, pushes = generic_instructions[instr.opname]

                arguments = stack[-reads:]
                arguments.reverse()

                stack = stack[:-pops]

                for i in range(pushes):
                    stack.append(GenericStackItem(opid, instr.opname, instr.argval, arguments, i))

            else:
                raise NotImplementedError("Can't construct graph from code object with instruction {}".format(instr))

        return graph

    def __json__(self, enc: util.JSONEncoder):
        """Return a JSON representable version of the graph."""
        edges = set()
        for fromid, e in enumerate(self._edges):
            for fromport, toid, toport in e._forward:
                edges.add((fromid, fromport, toid, toport))

        d = {"ops": [enc.typed(op) for op in self._ops],
             "edges": list(edges),
             "outputs": self._demands,
             "name": self.item_name}
        return d

    def __fromjson__(self, d: Any, dec: util.JSONDecoder):
        """Read a graph from a json object."""
        if 'ops' not in d or 'edges' not in d or 'outputs' not in d or \
           'name' not in d:
            raise util.JSONMissingKeys(type(self), ('ops', 'edges', 'outputs', 'names',))

        self.__init__(d['name'])

        for o in d['ops']:
            if isinstance(o, Param):
                self.add_param(o._name, o._type)
            else:
                self.add_op(o)

        for fromid, fromport, toid, toport in d['edges']:
            self.connect(fromid, toid, fromport=fromport, toport=toport)

        for name in d['outputs']:
            fromid, fromport = d['outputs'][name]
            self.demand(fromid, fromport=fromport, outname=name)

    def __call__(self, **kwargs):
        """Invoke the graph and get a graph instance."""
        return GraphInstance(self, params=kwargs)

    def perform(self):
        """Raise an error that graphs cannot be performed directly."""
        raise RuntimeError("Cannot run graphs directly")

    def get_signature(self):
        """Compute the graph's signature."""
        if self._signature is not None:
            return self._signature
        else:
            inputs = {}
            outputs = {}

            for p in self._ops:
                if isinstance(p, Param):
                    inp = Input(p._name, p._type)
                    inputs[p._name] = inp

            for nm in self._demands:
                fromid, fromport = self._demands[nm]
                output = self._ops[fromid].get_signature().outputs[fromport]

                out = Output(name=output.name, typedecl=output.type,
                             dynamic=output.dynamic,
                             metadata=output.metadata,
                             description=output.description)
                outputs[nm] = out

            return Signature(inputs=inputs, outputs=outputs)

InstanceId = NewType("InstanceId", str)

class GraphInstance(object):
    __slots__ = ("id", "executor", "graph",
                 "_params", "_complete", "_ready",
                 "_waiting", "_result_locations",
                 "_result_pointers")

    id: Optional[InstanceId]
    graph: Graph
    _params: Mapping[str, Any]
    _complete: set[tuple[int, str]]
    _ready: set[int]
    _waiting: list[set[tuple[int, str]]]
    _result_locations: dict[tuple[int, str], Storage]

    @staticmethod
    def allocate_id() -> InstanceId:
        return InstanceId(str(uuid.uuid4()))

    def __init__(self, graph: Graph, params: Mapping[str, Any]={}):
        """Create a new graph instance."""
        if not isinstance(graph, Graph):
            raise TypeError("graph must be a Graph object")

        self.id = None
        self.executor = None
        self._params = params
        self.graph = graph

        self._calculate_complete()
        self._calculate_waiting()

    def _calculate_complete(self):
        """Calculate the set of nodes that ought to be marked complete."""
        params = self._params
        graph = self.graph

        self._complete = set()
        for i, p in enumerate(graph._ops):
            if not isinstance(p, Param):
                continue

            if p._name not in params:
                raise TypeError(f"Missing parameter {p._name}")

            if not pytypes.is_of_type(params[p._name], p._type):
                raise TypeError("Type mismatch while applying {}: got {}, expected {}".format(p._name, type(params[p._name]), p._type))
            self._complete.add((i, "output"))

    def _calculate_waiting(self):
        """Calculate the set of dependencies."""
        graph = self.graph

        self._waiting = [set() for _ in range(len(graph._ops))]
        for opid, p in enumerate(graph._ops):
            if isinstance(p, OpDefinition):
                # Check if all inputs into this operation are defined. If so, add it to the ready queue
                self._waiting[opid] = {(fromid, fromport)
                                       for fromid, fromport, toport
                                       in graph._edges[opid]._backward
                                       if (fromid, fromport) not in self._complete}

    def bind_executor(self, executor: "dagger.executor.Executor"):
        """Bind the given executor."""
        self.executor = executor

    def bind(self, id: InstanceId, executor: "dagger.executor.Executor"):
        """Bind the instance to an id and an executor."""
        if self.is_bound:
            raise RuntimeError("Already bound")

        self.id = id
        self.bind_executor(executor)

        self._result_locations = {}
        # Decide where to store everything
        for opid, operation in enumerate(self.graph._ops):
            if isinstance(operation, Param):
                continue

            outputs = operation.get_signature().outputs
            default_storage = executor.context.storage
            for output_name in outputs:
                output = outputs[output_name]
                storage = getattr(output, 'storage', default_storage)
                self._result_locations[(opid, output_name)] = storage

    @property
    def is_bound(self):
        """Return whether this node has an ID (i.e, has been set for execution)."""
        return self.id is not None

    def ready(self):
        """Generate the set of ready nodes."""
        for opid, p in enumerate(self._waiting):
            op = self.graph._ops[opid]
            if len(p) == 0 and isinstance(op, OpDefinition):
                yield (opid, op)

    def mark_complete(self, nodeid, nodeport="output"):
        """Mark the given port as complete."""
        if nodeid > len(self.graph._ops):
            raise KeyError(nodeid)

        op = self.graph._ops[nodeid]
        if isinstance(op, Param):
            return

        if nodeport not in op.get_signature()._outputs:
            raise KeyError((nodeid, nodeport))

        self._complete.add((nodeid, nodeport))

        for (fromport, tonode, toport) in self.graph._edges[nodeid]._forward:
            if fromport != nodeport:
                continue

            self._waiting[tonode].discard((nodeid, nodeport))

    def is_complete(self, nodeid, nodeport="output"):
        """Check if the given port is complete."""
        return (nodeid, nodeport) in self._complete

    def all_outputs_complete(self):
        """Check if all outputs have been generated."""
        sig = self.graph.get_signature()
        for output_name in sig.outputs:
            fromid, fromport = self.graph.get_demand(output_name)
            if not self.is_complete(fromid, nodeport=fromport):
                return False
        return True

    def get_output_location(self, nodeid: int, portname: str) -> Storage:
        """Get the Storage that stores this result value."""
        if (nodeid, portname) not in self._result_locations:
            raise KeyError((nodeid, portname))
        return self._result_locations[(nodeid, portname)]

    def _prepare_arguments(self, nodeid: int) -> tuple[list[Storage],
                                                       Iterable[InputLocation],
                                                       Mapping[str, InputLocation],
                                                       Mapping[str, int]]:
        """Prepare arguments for submission to the execution layer."""
        node = self.graph._ops[nodeid]
        if isinstance(node, OpDefinition):
            sig = node.get_signature()
            inputs = {}

            param_storage = MemoryStorage()

            for fromid, fromport, toport in self.graph._edges[nodeid]._backward:
                fromop = self.graph._ops[fromid]
                if isinstance(fromop, Param):
                    location = param_storage.store_result(None, None,
                                                          {'instance': self.id,
                                                           'port': fromop._name,
                                                           'op': None},
                                                          self._params[fromop._name])
                    inputs[toport] = InputLocation(storage=param_storage, location=location)
                else:
                    storage: Storage = self._result_locations[(fromid, fromport)]
                    op_result = self.executor.context.state.load_op_result(self.id, fromid, fromport)

                    inputs[toport] = InputLocation(storage=storage, location=op_result)

            for inp_name in sig.inputs:
                if inp_name not in inputs:
                    raise RuntimeError(f'Input {inp_name} missing and not caught during instance creation')

            # Calculate outputs
            outputs = {}
            for o in sig.outputs:
                outputs[o] = self._result_locations[(nodeid, o)]

            # Calculate final values

            storages: list[Storage] = []
            storage_unique: dict[int, int] = {}
            for input_name in inputs:
                inp = inputs[input_name]
                if id(inp.storage) in storage_unique:
                    storage_id = storage_unique[id(inp.storage)]
                else:
                    storage_id = storage_unique[id(inp.storage)] = len(storages)
                    storages.append(inp.storage)

                inputs[input_name] = InputLocation(storage=storage_id, location=inp.location)

            output_locations = {}
            for output_name in outputs:
                out = outputs[output_name]
                if id(out) in storage_unique:
                    output_locations[output_name] = storage_unique[id(out)]
                else:
                    output_locations[output_name] = storage_unique[id(out)] = len(storages)
                    storages.append(out)

            args = []
            kwargs: dict[str, InputLocation] = {}
            for inp_name in node.get_input_order():
                args.append(inputs[inp_name])

            return (storages, args, kwargs, output_locations)

        else:
            raise TypeError("Arguments can only be prepared for ops")


    def __getitem__(self, nodeid):
        """Return the node associated with the id."""
        return self.graph._ops[nodeid]

    def __json__(self, e: util.JSONEncoder):
        """Serialize this instance into a JSON-serializable struct."""
        return {"id": self.id,
                "graph": self.graph,
                "params": self._params,
                "complete": [x for x in self._complete],
                "result_locations": [((nodeid, port), e.typed(s))
                                     for (nodeid, port), s in self._result_locations.items()]}

    def __fromjson__(self, d: Any, dec: util.JSONDecoder):
        """Read the instance from a JSON document."""
        if 'id' not in d or 'graph' not in d or 'params' not in d or \
           'complete' not in d or 'result_locations' not in d:
            raise util.JSONMissingKeys(type(self), ('id', 'graph', 'params',
                                                    'complete', 'result_locations',))

        self.id = d['id']
        self.graph = dec.as_(d['graph'], Graph)
        self._params = d['params']
        self._result_locations = { (nodeid, port): s for (nodeid, port), s in d['result_locations'] }
        self.executor = None

        self._calculate_complete()
        self._calculate_waiting()

        for nodeid, port in d['complete']:
            self.mark_complete(nodeid, port)


def graph(fn):
    """Turn a function into a graph"""
    g = Graph.from_function(fn)
    g.__wrapped__ = fn

    g.validate()

    return g

@op
def op1(test: str) -> str:
    return "op1({})".format(test)

@op
def op2(a: str) -> str:
    return "op2({})".format(a)

@op
def op3(a: str) -> str:
    return "op3({})".format(a)

@op
def op4(a: str, b: str) -> str:
    return "op4({}, {})".format(a, b)

@op
def op5(a: str) -> str:
    return "op5({})".format(a)

@op
def op6(a: str, b:str) -> str:
    return "op6({}, {})".format(a, b)

def test_fn(param1: str) -> str:
    a = op1(param1)
    a1 = op2(a)
    a2 = op3(a)
    a3 = op4(a1, a2)
    a5 = op5(param1)
    return op6(a3, a5)

@graph
def test_graph(param1: str) -> str:
    a = op1(param1)
    a1 = op2(a)
    a2 = op3(a)
    a3 = op4(a1, a2)
    a5 = op5(param1)
    return op6(a3, a5)

