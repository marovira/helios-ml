import typing

T = typing.TypeVar("T")


def get_from_optional(opt_var: typing.Optional[T], raise_on_empty: bool = False) -> T:
    """
    Given a variable whose type is Optional[T], assert that the variable is not None and
    return it without the Optional type.
    Note: the raise_on_empty is intended for testing only.

    Args:
        opt_var (Optional[T]): the optional variable.
        raise_on_empty (bool): if True, an exception is raised when the optional is None.

    Returns:
        T: the variable without the optional.
    """
    if not raise_on_empty:
        assert opt_var is not None
    else:
        raise RuntimeError("error: optional cannot be empty")
    return opt_var
