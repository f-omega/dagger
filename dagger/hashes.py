"""Provides common hashes for use with operation outputs."""
from typing import TypedDict, Optional, Sequence, Mapping, Any
from io import BytesIO
import pickle
import base64


class ProvenanceInfo(TypedDict, total=False):
    """Used to tell the storage where a result was taken from."""

    instance: "dagger.graph.InstanceId"
    op: Optional[int]
    opname: Optional[str]
    port: Optional[str]
    order: Optional[int]

    args: Optional[Sequence[Any]]
    kwargs: Optional[Mapping[str, Any]]


def default(op: "dagger.op.OpDefinition", prov: ProvenanceInfo, out: BytesIO):
    """Hash a result by a normalized representation of its arguments."""

    # Write out the operation name
    out.write("{}\n".format(op.item_name).encode('utf-8'))

    # At the end of the day, every argument is a keyword argument
    # Each keyword argument is written, along with its name on one line
    args = {name: None for name in op.get_default_args()}
    arglocs = prov['args'] or []
    for argnm, argloc in zip(op.get_input_order(), arglocs):
        args[argnm] = argloc

    kwarglocs = prov['kwargs'] or {}
    for argnm, argloc in kwarglocs.items():
        args[argnm] = argloc

    sorted_arg_nms = list(args.keys())
    sorted_arg_nms.sort()

    for argnm in sorted_arg_nms:
        argloc = args[argnm]

        dumped_file = BytesIO()
        pickle.dump(argloc, dumped_file)

        dumped_file.seek(0)

        b64_file = BytesIO()
        base64.encode(dumped_file, b64_file)

        # Remove newlines
        dumped = b64_file.getvalue().replace(b'\n', b'')

        out.write(argnm.encode('utf-8'))
        out.write(b' ')
        out.write(dumped)
        out.write(b'\n')

    return out.getvalue()
