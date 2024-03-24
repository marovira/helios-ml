import os
import pathlib
import time
import typing

T = typing.TypeVar("T")
T_Any = typing.TypeVar("T_Any", bound=typing.Any)


def get_from_optional(opt_var: T | None, raise_on_empty: bool = False) -> T:
    """
    Ensure the given variable is not None and return it.

    This is useful when dealing with variables that can be None at declaration but are set
    elsewhere. In those instances, Mypy is unable to determine that the variable was set,
    so it will issue a warning. The workaround is to add asserts, but that can get tedious
    very quickly. This function can be used as an alternative.
    Ex:
        ```py
        var: int | None = None
        # ... Set var to a valid value some place else.

        assert var is not None
        v = var

        # Alternatively:
        v = core.get_from_optional(var)
        ```

    Args:
        opt_var (T | None): the optional variable.
        raise_on_empty (bool): if True, an exception is raised when the optional is None.

    Returns:
        T: the variable without the optional.
    """
    if not raise_on_empty:
        assert opt_var is not None
    else:
        raise RuntimeError("error: optional cannot be empty")
    return opt_var


def convert_to_list(var: T | list[T] | tuple[T, ...]) -> list[T]:
    """
    Convert the input into a list if it's not one already.

    This is a utility that replaces the following pattern:
        ```py
        def some_fun(x: int | list[int]) -> None:
            if isinance(x, list):
                x = [x]
            for elem in x:
                ...

            # The above code an be replaced with this:
            for elem in convert_to_list(x):
                ...
        ```

    Args:
        var (T : list[T]): an object that can be either a single object or a list.

    Returns:
        list[T]: if the input was a list, no operation is done. Otherwise, the object is
        converted to a list and returned.
    """
    if isinstance(var, list):
        return var
    if isinstance(var, tuple):
        return list(var)
    return [var]


class ChdirContext:
    """
    Allow switching between the current working directory and another within a scope.

    The intention is to facilitate temporary switches of the current working directory
    (such as when attempting to resolve relative paths) by creating a context in which the
    working directory is automatically switched to a new one. Upon exiting of the context,
    the original working directory is restored.

    Ex:
        ```py
        os.chdir(".")   # <- Starting working directory
        with ChdirContext("/new/path") as prev_cwd:
            # prev_cwd is the starting working directory
            Path.cwd() # <- This is /new/path now
            ...
        Path.cwd() # <- Back to the starting working directory.
        ```

    Args:
        target_path (pathlib.Path): the path to switch to.
    """

    def __init__(self, target_path: pathlib.Path):
        """
        Create the context manager with the given path.

        Args:
            target_path (pathlib.Path): the path to switch to.
        """
        self.start_path = pathlib.Path.cwd()
        self.target_path = target_path

    def __enter__(self) -> pathlib.Path:
        """
        Perform the switch from the current working directory to the new one.

        Returns:
            pathlib.Path: the previous working directory.
        """
        os.chdir(self.target_path)
        return self.start_path

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Restores the previous working directory."""
        os.chdir(self.start_path)


class AverageTimer:
    """
    Compute elapsed times using moving average.

    The timer will determine the elapsed time between a series of points using a sliding
    window moving average.

    Args:
        sliding_window (int): the number of steps of which the moving average will be
        computed.
    """

    def __init__(self, sliding_window: int = 200):
        """
        Create the timer with the given sliding window.

        Args:
            sliding_window (int): the number of steps of which the moving average will be
            computed.
        """
        self._sliding_window = sliding_window

        self._time_sum: float = 0
        self._step_count: int = 0
        self._current_time: float = 0
        self.start()

    def start(self) -> None:
        """Start the timer."""
        self._current_time = time.time()

    def record(self) -> None:
        """Record a new step in the timer."""
        self._step_count += 1
        self._time_sum += time.time() - self._current_time

        if self._step_count > self._sliding_window:
            self._step_count = 0
            self._time_sum = 0

        self._current_time = time.time()

    def get_average_time(self) -> float:
        """Return the moving average over the current step count."""
        return self._time_sum / self._step_count


class Registry:
    """
    Provides a name -> object mapping to allow users to create custom types.

    Usage:
        ```py
        # Create a registry:
        TEST_REGISTRY = Registry("test")

        # Register as a decorator:
        @TEST_REGISTRY.register
        class TestClass:
            ...

        # Register in code:
        TEST_REGISTRY.register(TestClass)
        TEST_REGISTRY.register(test_function)
        ```
    Args:
        name (str): the name of the registry.
    """

    def __init__(self, name: str):
        """
        Create the registry with the given name.

        Args:
            name (str): the name of the registry.
        """
        self._name = name
        self._obj_map: dict[str, typing.Any] = {}

    def _do_register(self, name: str, obj: typing.Any, suffix: str | None = None) -> None:
        """
        Register the function/class.

        Args:
            name (str): the name of the object to register.
            obj (Any): the object to register.
            suffix (str | None): an optional suffix to add to the name upon
            registration.
        """
        if isinstance(suffix, str):
            name = name + "_" + suffix

        assert (
            name not in self._obj_map
        ), f"error: an object named '{name}' already exists in the "
        f"'{self._name}' registry"

        self._obj_map[name] = obj

    def register(self, obj: T_Any, suffix: str | None = None) -> T_Any:
        """
        Register the given object.

        Args:
            obj (T_Any): the type to add. Must have a __name__ attribute.
            suffix (str | None): the suffix to add to the type name.

        Returns:
            T_Any: the registered type.
        """
        name = obj.__name__
        self._do_register(name, obj, suffix)
        return obj

    def get(self, name: str, suffix: str | None = None) -> typing.Any:
        """
        Get the object that corresponds to the given name.

        Args:
            name (str): the name of the type.
            suffix (str): the suffix to use if the type isn't found with the given name.

        Returns:
            Any: the requested type.
        """
        ret = self._obj_map.get(name)
        if ret is None and suffix is not None:
            name_suff = name + "_" + suffix
            ret = self._obj_map.get(name_suff)
            print(f"warning: found {name_suff} instead of {name}")
            if ret is None:
                raise KeyError(
                    f"No object called '{name}' found in the '{self._name}' registrar"
                )
        return ret

    def __contains__(self, name: str) -> bool:
        """
        Check if the registry contains the given name.

        Args:
            name (str): the name to check.

        Returns:
            bool: true if the name exists, false otherwise.
        """
        return name in self._obj_map

    def __iter__(self) -> typing.Iterable:
        """Get an iterable over the registry items."""
        return iter(self._obj_map.items())

    def __str__(self) -> str:
        """Get the name of the registry."""
        return self._name

    def keys(self) -> typing.Iterable:
        """
        Return a set-like object providing a view into the registry's keys.

        Return:
            Iterable: an iterable of the registry keys.
        """
        return self._obj_map.keys()
