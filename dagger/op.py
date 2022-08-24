'''Ops are pieces of code that run atomically, either producing a result or not.

Ops have several properties:

* **executor** - The executor used to execute this operation by default
* **store** - How the result of this operation is persisted during the graph execution
* **dependencies** - List of artifacts, ops, graphs, etc that this operation depends on
'''

from typing import Callable, Iterable, Any, Optional, \
    Union, Mapping, TypedDict, Dict, List, Tuple, Iterator, BinaryIO
from abc import abstractmethod
from inspect import signature, isgeneratorfunction
from collections import namedtuple
from functools import partial

from .repo import Repository, RepositoryItem, ItemName, \
    ItemType, get_global_repository
from . import hashes
from .hashes import ProvenanceInfo


class DynamicOutputAnnotated(Exception):
    """Thrown when a dynamic output was specified but an annotation was noted."""

    def __init__(self, callable):
        """Initialize the exception."""
        self._callable = callable

    def __str__(self):
        """Get the exception as a string."""
        return "Dynamic output specified in function with annotation"


PortResult = namedtuple('PortResult', ('port', 'data', 'complete',),
                        defaults=(False,))


class InputSpec(TypedDict, total=False):
    """TypedDict for input."""

    typedecl: type
    name: str
    description: Optional[str]
    metadata: Mapping[str, str]


class OutputSpec(TypedDict, total=False):
    """TypedDict for output."""

    typedecl: type
    name: str
    description: Optional[str]
    metadata: Mapping[str, str]
    dynamic: bool


class Input(object):
    """Represents an input into an Op."""

    __slots__ = ('_name', '_description',
                 '_metadata', '_type',
                 '_has_default', '_default')

    def __init__(self, name, typedecl, description=None,
                 metadata: Mapping[str, str] = {},
                 has_default: bool = False,
                 default: Any = None):
        """Initialize an input from parameters."""
        self._name = name
        self._type = typedecl
        self._description = description
        self._metadata = metadata
        self._has_default = has_default
        self._default = default

    def __eq__(self, o):
        """Check if two inputs are equal by comparing names and types."""
        return isinstance(o, Input) and \
            o._name == self._name and \
            o._type == self._type


class Output(object):
    """Represents an output from an Op."""

    __slots__ = ('name', 'description',
                 'metadata', 'type', '_dynamic',)

    @property
    def dynamic(self):
        """Return whether the output is dynamic (has multiple values) or not."""
        return self._dynamic

    def __init__(self, name, typedecl,
                 description: Optional[str] = None,
                 dynamic: bool = False,
                 metadata: Mapping[str, str] = {}):
        """Initialize an output from parameters."""
        self.name = name
        self.type = typedecl
        self.description = description
        self._dynamic = dynamic
        self.metadata = metadata

    def __eq__(self, o):
        """Check if the other output is equal by name, type, and dynamicity."""
        return isinstance(o, Output) and \
            self.name == o._name and \
            self.type == o._type and \
            self._dynamic == o._dynamic


class OpDefinition(RepositoryItem):
    """An operation in a repository."""

    __slots__ = ('_name', '_mkhashinfo',)

    _mkhashinfo: Callable[[ProvenanceInfo, BinaryIO], None]

    def __init__(self, name,
                 mkhashinfo: Callable[["OpDefinition", ProvenanceInfo, BinaryIO], None]):
        """Initialize an operation given a name."""
        super().__init__()

        self._name = name
        self._mkhashinfo = partial(mkhashinfo, self)

    def get_item_dependencies(self):
        """Return the dependencies for this op."""
        return []

    @property
    def item_name(self) -> ItemName:
        """Get the name of the operation."""
        return self._name

    @classmethod
    def get_item_type(self) -> ItemType:
        """Get the type of the operation.

        :returns "op"
        """
        return ItemType("op")

    @abstractmethod
    def perform(self, *args, **kwargs) -> Iterator[PortResult]:
        """Invoke the operation to occur."""
        pass

    @abstractmethod
    def get_signature(self) -> "Signature":
        """Get the inputs into this op."""
        pass

    def get_input_order(self) -> Iterable[str]:
        """Get the inputs in the order expected. Used to resolve arguments."""
        return [i._name for i in self.get_signature()._inputs.values()]

    def get_default_args(self) -> Mapping[str, Any]:
        """Get the default arguments passed to this operation if the
           arg with this name is not given.

           If an argument does not appear in this mapping, then it is required.
        """
        return {i._name: i._default for i in self.get_signature()._inputs.values()
                if i._default is not None}


class Signature(object):
    """Represents the signature of an op."""

    __slots__ = ('_inputs', '_outputs', '_dynamic',)

    _inputs: Mapping[str, Input]
    _outputs: Mapping[str, Output]

    def __init__(self, inputs: Mapping[str, Input] = {},
                 outputs: Mapping[str, Output] = {}):
        """Initialize a signature from input and output declarations."""
        self._inputs = inputs
        self._outputs = outputs

        self._dynamic = any(outputs[o]._dynamic for o in outputs)

    @property
    def outputs(self):
        """Get the outputs from this op."""
        return self._outputs

    @property
    def inputs(self):
        """Get the inputs to this op."""
        return self._inputs


class ConstantOpDefinition(OpDefinition):
    """An operation that returns a constant."""

    __slots__ = ("_constant",)

    def __init__(self, c):
        """Initialize the op with the given constant."""
        super().__init__("constant_{}".format(repr(c)))
        self._constant = c

    def perform(self, *args, **kwargs) -> Any:
        """Return the constant."""
        return self._constant

    def get_signature(self) -> Signature:
        """Get the signature of this operation."""
        sig = Signature(outputs={'output': Output('output', type(self._constant))})
        return sig


class CallableOpDefinition(OpDefinition):
    """An operation that is defined via a callable."""

    __slots__ = ('_callable', '_signature', '_is_generator', '_extract_ports')

    _extract_ports: dict[str, Callable[[Any], Any]]

    def __init__(self, name: str, callable: Callable,
                 mkhashinfo: Callable[[OpDefinition, ProvenanceInfo, BinaryIO], None] = hashes.default,
                 signature: Optional[Signature] = None,
                 inputs: Optional[List[Union[Input, InputSpec]]] = None,
                 outputs: Optional[List[Union[Output, OutputSpec]]] = None,
                 repo: Optional[Repository] = None):
        """Initialize the operation from a name and callable."""
        super(CallableOpDefinition, self).__init__(name, mkhashinfo)
        self._callable = callable
        self._is_generator = isgeneratorfunction(callable)
        self._extract_ports = {}

        if repo is None:
            repo = get_global_repository()

        if signature is None:
            inputs_dict: Dict[str, Union[Input, InputSpec]] = {}
            if inputs is None:
                inputs = []

            for i in inputs:
                if isinstance(i, Input):
                    inputs_dict[i._name] = i
                elif 'name' in i:
                    inputs_dict[i['name']] = i
                else:
                    raise TypeError("Inputs must be named")
            self._signature = self._validate(inputs_dict, outputs)
        elif inputs is None and outputs is None:
            self._signature = signature
        else:
            raise TypeError("Cannot provide signatures and either inputs or outputs")

        repo.add(self)

    def __repr__(self):
        """Return a machine-readable version of the definition."""
        return "<CallableOpDefinition {}>".format(self._callable.__name__)

    def __json__(self, e):
        """Get the json representation of this op."""
        return e.pointer(module=self._callable.__module__,
                         name=self._callable.__name__,
                         original=self)

    def _validate(self, inputs: Optional[Mapping[str,
                                                 Union[Input, InputSpec]]],
                  outputs: Optional[List[Union[Output, OutputSpec]]])\
                  -> Signature:
        """Validate that the callable's type signature."""
        sig = signature(self._callable)

        ann_inputs: Dict[str, Input] = {}

        for param_name in sig.parameters:
            param = sig.parameters[param_name]
            ann = param.annotation

            if ann == param.empty:
                raise TypeError("Must be annotated with proper type to be an op")

            # Check if there's a spec given
            if inputs is not None and \
               param_name in inputs:
                spec = inputs[param_name]
                if isinstance(spec, Input):
                    ann_inputs[param_name] = spec
                else:
                    ann = spec.get('typedecl', ann)

                    new_input_spec: Any = dict(**spec)
                    new_input_spec['name'] = param_name
                    new_input_spec['typedecl'] = ann

                    spec = Input(**new_input_spec)
                    ann_inputs[param_name] = spec
            else:
                ann_inputs[param_name] = Input(name=param_name, typedecl=ann)

        ann_outputs: Dict[str, Output] = {}
        if sig.return_annotation != sig.empty:
            if hasattr(sig.return_annotation, '__origin__') and \
               (sig.return_annotation.__origin__ is Tuple or \
                sig.return_annotation.__origin__ is tuple):
                ann_outs = sig.return_annotation.__args__

                if outputs is None:
                    raise TypeError("When returning multiple outputs, you must specify an output annotation")
                elif len(outputs) == 1:
                    if not isinstance(outputs[0], Output) and \
                       'name' not in outputs[0]:
                        outputs[0]['name'] = 'output'

                if len(outputs) != len(ann_outs):
                    raise TypeError('There are more declared outputs than annotated ones')

                for output_idx, output, ann in enumerate(zip(outputs, ann_outs)):
                    # Reconcile with output types
                    if isinstance(output, Output):
                        ann_outputs[output.name] = output
                    else:
                        if 'name' not in output:
                            raise TypeError("output names must be declared")

                        output_spec: Any = dict(**output)
                        output_spec['typedecl'] = output_spec.get('typedecl', ann)

                        output = Output(**output_spec)
                        ann_outputs[output_spec['name']] = output
                        self._extract_ports[output_spec['name']] = lambda x: x[output_idx]

                    if output.dynamic:
                        raise DynamicOutputAnnotated(self._callable)
            else:
                template = None
                if outputs is not None and len(outputs) == 1:
                    template = outputs[0]
                elif outputs is not None:
                    raise TypeError("Too many outputs given")

                if template is not None:
                    if isinstance(template, Output):
                        if template.type != sig.return_annotation:
                            raise TypeError("Output type and annotated type don't match")
                        output = template
                    else:
                        new_template: Any = dict(**template)
                        new_template['name'] = 'output'
                        new_template['typedecl'] = sig.return_annotation
                        output = Output(**new_template)
                else:
                    output = Output(name='output', typedecl=sig.return_annotation)

                ann_outputs['output'] = output
                self._extract_ports[output.name] = lambda x: x

                if output.dynamic:
                    raise DynamicOutputAnnotated(self._callable)
        elif outputs is not None:
            for o in outputs:
                if isinstance(o, Output):
                    ann_outputs[o.name] = o
                else:
                    ann_outputs[o['name']] = Output(**o)
        else:
            raise TypeError("A return type must be specified or outputs given")

        return Signature(inputs=ann_inputs, outputs=ann_outputs)

    def perform(self, *args, **kwargs) -> Iterator[PortResult]:
        """Perform the operation by calling the callable."""
        if self._is_generator:
            sig = self.get_signature()

            if len(sig.outputs) == 1:
                def_output = sig.outputs.keys()[0]
            else:
                def_output = None

            for x in self._callable(*args, **kwargs):
                if isinstance(x, PortResult):
                    yield x
                elif def_output is not None:
                    yield PortResult(port=def_output, data=x)
        else:
            out = self._callable(*args, **kwargs)
            for port_name, get_port in self._extract_ports.items():
                port_value = get_port(out)
                yield PortResult(port=port_name, data=port_value)

    def get_signature(self) -> Signature:
        """Get signature for this op."""
        return self._signature

    def get_input_order(self) -> Iterable[str]:
        """Return the order of the callable's arguments."""
        sig = signature(self._callable)

        return [p for p in sig.parameters]

    @property
    def __wrapped__(self):
        """Return the wrapped callable."""
        return self._callable

    def __call__(self, *args, **kwargs):
        """Throw an error indicating that ops are not callable."""
        raise TypeError("Ops can only be called from within a graph")

def op(fn: Optional[Callable] = None, **kwargs) -> Union[CallableOpDefinition, Callable[[Callable], CallableOpDefinition]]:
    """Decorator to quickly define an op."""
    if fn is not None:
        return CallableOpDefinition(fn.__name__, fn)
    else:
        def op_closure(fn: Callable) -> CallableOpDefinition:
            return CallableOpDefinition(fn.__name__, fn, **kwargs)
        op_closure.__name__ = "op"
        return op_closure
