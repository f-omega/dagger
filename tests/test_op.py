import pytest

from dagger import op, Input, Output, DynamicOutputAnnotated

def test_simple_op():
    @op
    def simple_op(a: int, b: int) -> int:
        return a + b

    assert simple_op.get_signature()._inputs == {"a": Input("a", int),
                                                 "b": Input("b", int)}
    assert simple_op.get_signature()._outputs == {"output": Output("output", int)}

def test_dynamic_op_with_return_annotation():
    with pytest.raises(DynamicOutputAnnotated):
        @op(outputs=[Output("output", typedecl=int, dynamic=True)])
        def simple_op(a: int, b: int) -> int:
            yield a + b
            yield a
            yield b

def test_dynamic_op():
    @op(outputs=[Output("output", typedecl=int, dynamic=True)])
    def simple_op(a: int, b: int):
        yield a + b
        yield a
        yield b

    assert simple_op.get_signature()._inputs == {"a": Input("a", int),
                                                 "b": Input("b", int)}
    assert simple_op.get_signature()._outputs == {"output": Output("output", int, dynamic=True)}

def test_multiout_op():
    @op(outputs=[Output("a", typedecl=int),
                 Output("b", typedecl=str)])
    def simple_op(a: int, b: int) -> tuple[int, str]:
        return a, str(b)

    assert simple_op.get_signature()._inputs == {"a": Input("a", int),
                                                 "b": Input("b", int)}
    assert simple_op.get_signature()._outputs == {"a": Output("a", int),
                                                  "b": Output("b", str)}

def test_multiout_op_bad_annotation():
    with pytest.raises(TypeError):
        @op(outputs=[Output("a", typedecl=int),
                     Output("b", typedecl=str)])
        def simple_op(a: int, b: int) -> int:
            return a, str(b)
