import helios.core as hlc

FUNC_REGISTRY = hlc.Registry("func")


def create_func(type_name: str):
    return FUNC_REGISTRY.get(type_name)
