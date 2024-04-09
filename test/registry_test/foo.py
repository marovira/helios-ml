from .func_registry import FUNC_REGISTRY


@FUNC_REGISTRY.register
def foo() -> str:
    return "foo"
