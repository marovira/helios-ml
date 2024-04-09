import pyro.core as pyc

FUNC_REGISTRY = pyc.Registry("func")


def create_func(type_name: str):
    return FUNC_REGISTRY.get(type_name)
