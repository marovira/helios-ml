import importlib
import os
import pathlib
import platform
import re
import sys
import time
import types
import typing

import torch
import torchvision

from .._version import __version__

T = typing.TypeVar("T")
T_Any = typing.TypeVar("T_Any", bound=typing.Any)


def get_env_info_str() -> str:
    """
    Return a string with the Helios header and the environment information.

    Returns:
        The message string.
    """
    msg = r"""
#===========================================================================#
          _______  _       _________ _______  _______
|\     /|(  ____ \( \      \__   __/(  ___  )(  ____ \
| )   ( || (    \/| (         ) (   | (   ) || (    \/
| (___) || (__    | |         | |   | |   | || (_____
|  ___  ||  __)   | |         | |   | |   | |(_____  )
| (   ) || (      | |         | |   | |   | |      ) |
| )   ( || (____/\| (____/\___) (___| (___) |/\____) |
|/     \|(_______/(_______/\_______/(_______)\_______)
    """
    msg += (
        "\nEnvironment info: "
        f"\n\tHelios: {__version__}"
        f"\n\tPyTorch: {torch.__version__}"
        f"\n\tTorchVision: {torchvision.__version__}"
        f"\n\tOS: {platform.platform()}"
        f"\n\tPython: {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[3]}"
    )
    if torch.cuda.is_available():
        msg += f"\n\tCUDA version: {torch.version.cuda}"
    msg += "\n"
    msg += "#===========================================================================#"
    msg += "\n\n"
    return msg


def get_from_optional(opt_var: T | None, raise_on_empty: bool = False) -> T:
    """
    Ensure the given variable is not :code:`None` and return it.

    This is useful when dealing with variables that can be ``None`` at declaration but
    are set elsewhere. In those instances, mypy is unable to determine that the variable
    was set, so it will issue a warning. The workaround is to add asserts, but that can
    get tedious very quickly. This function can be used as an alternative.

    Example:
        .. code-block:: python

            var: int | None = None
            # ... Set var to a valid value some place else.

            assert var is not None
            v = var

            # Alternatively:
            v = core.get_from_optional(var)


    Args:
        opt_var: the optional variable.
        raise_on_empty: if True, an exception is raised when the optional is ``None``.

    Returns:
        The variable without the optional.

    Raises:
          RuntimeError: if the ``opt_var`` is ``None`` and ``raise_on_empty`` is true.

    """
    if not raise_on_empty:
        assert opt_var is not None
    else:
        raise RuntimeError("error: optional cannot be empty")
    return opt_var


def convert_to_list(var: T | list[T] | tuple[T, ...]) -> list[T]:
    """
    Convert the input into a list if it's not one already.

    Example:
        .. code-block:: python

            def some_fun(x: int | list[int]) -> None:
                if isinstance(x, list):
                    x = [x]
                for elem in x:
                    ...

                # The above code an be replaced with this:
                for elem in convert_to_list(x):
                    ...

    Args:
        var: an object that can be either a single object or a list.

    Returns:
        If the input was a list, no operation is done. Otherwise, the object is converted
        to a list and returned.
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

    Example:
        .. code-block:: python

            os.chdir(".")   # <- Starting working directory
            with ChdirContext("/new/path") as prev_cwd:
                # prev_cwd is the starting working directory
                Path.cwd() # <- This is /new/path now
                ...
            Path.cwd() # <- Back to the starting working directory.

    Args:
        target_path: the path to switch to.
    """

    def __init__(self, target_path: pathlib.Path):
        """
        Create the context manager with the given path.

        Args:
            target_path: the path to switch to.
        """
        self.start_path = pathlib.Path.cwd()
        self.target_path = target_path

    def __enter__(self) -> pathlib.Path:
        """
        Perform the switch from the current working directory to the new one.

        Returns:
            The previous working directory.
        """
        os.chdir(self.target_path)
        return self.start_path

    def __exit__(
        self,
        exc_type: type[Exception] | None,
        exc_value: Exception | None,
        exc_traceback: types.TracebackType | None,
    ) -> None:
        """Restores the previous working directory."""
        os.chdir(self.start_path)


class AverageTimer:
    """
    Compute elapsed times using moving average.

    The timer will determine the elapsed time between a series of points using a sliding
    window moving average.

    Args:
        sliding_window: number of steps over which the moving average will be computed.
    """

    def __init__(self, sliding_window: int = 200):
        """
        Create the timer with the given sliding window.

        Args:
            sliding_window: steps over which the moving average will be computed.
        """
        self._sliding_window = sliding_window

        self._time_sum: float = 0
        self._step_count: int = 0
        self._current_time: float = 0
        self._avg_time: float = 0
        self.start()

    def start(self) -> None:
        """Start the timer."""
        self._current_time = time.time()

    def record(self) -> None:
        """Record a new step in the timer."""
        self._step_count += 1
        self._time_sum += time.time() - self._current_time
        self._avg_time = self._time_sum / self._step_count

        if self._step_count > self._sliding_window:
            self._step_count = 0
            self._time_sum = 0

        self._current_time = time.time()

    def get_average_time(self) -> float:
        """Return the moving average over the current step count."""
        return self._avg_time


class Registry:
    """
    Provides a name to object mapping to allow users to create custom types.

    Example:
        .. code-block:: python

            # Create a registry:
            TEST_REGISTRY = Registry("test")

            # Register as a decorator:
            @TEST_REGISTRY.register
            class TestClass:
                ...

            # Register in code:
            TEST_REGISTRY.register(TestClass)
            TEST_REGISTRY.register(test_function)

    Args:
        name: the name of the registry.
    """

    def __init__(self, name: str):
        """
        Create the registry with the given name.

        Args:
            name: the name of the registry.
        """
        self._name = name
        self._obj_map: dict[str, typing.Any] = {}

    def _do_register(self, name: str, obj: typing.Any, suffix: str | None = None) -> None:
        """
        Register the function/class.

        Args:
            name: the name of the object to register.
            obj: the object to register.
            suffix: (optional) suffix to add to the name upon registration.
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
            obj: the type to add. Must have a __name__ attribute.
            suffix: (optional) the suffix to add to the type name.

        Returns:
            The registered type.
        """
        name = obj.__name__
        self._do_register(name, obj, suffix)
        return obj

    def get(self, name: str, suffix: str | None = None) -> typing.Any:
        """
        Get the object that corresponds to the given name.

        Args:
            name: the name of the type.
            suffix: (optional) the suffix to use if the type isn't found with the given
                name.

        Returns:
            The requested type.

        Raises:
            KeyError: if no object with the given name is found in the registry.
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
        elif ret is None:
            raise KeyError(
                f"No object called '{name}' found in the '{self._name}' registrar"
            )
        return ret

    def __contains__(self, name: str) -> bool:
        """
        Check if the registry contains the given name.

        Args:
            name: the name to check.

        Returns:
            True if the name exists, false otherwise.
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
            An iterable of the registry keys.
        """
        return self._obj_map.keys()


def update_all_registries(
    root: pathlib.Path, recurse: bool = True, import_prefix: str = ""
) -> None:
    """
    Ensure all registered types get added to their corresponding registries.

    This function serves as a way of automatically registering all types into their
    corresponding registries within a package. Normally, you'd have to manually include
    each module that contains a registered type to ensure that it gets registered. This
    can easily cascade if modules are nested inside packages, whereby the top-level module
    has to (somehow) ensure that all child modules get imported to ensure everything works
    correctly.

    This function offers an alternative, whereby it will automatically scan all
    modules and sub-packages within a given package and import only those files that
    register a type. To do this, there are a few assumptions:

    #. Each package MUST contain an ``__init__.py`` (namespace packages are not
       supported)
    #. A module is included if and only if there is at least one line that contains the
       following pattern: ``@<any non-whitespace character(s)>.register``.

    Example:
        Suppose we have a project with the following structure:

        .. code-block:: text

            main.py
            my_package/
            |---__init__.py
            |---some_class.py <- This registers a type.
            |---some_funcs.py <- Doesn't register anything.
            |---sub_package/
            |   |---__init__.py
            |   |---another_type.py <- Registers
            |   |---another_func.py <- Doesn't register.

        We can then do the following inside ``main.py``:

        .. code-block:: python

            import helios.core as hlc
            ...
            hlc.update_all_registries(Path.cwd() / "my_package", recurse=True)

        The function will recursively walk through ``my_package`` and import the
        following:

        * ``my_package.some_class``
        * ``my_package.sub_package.another_type``

        After the function returns, the corresponding registries will have been populated
        with the types and they can be used elsewhere in the code.

    Args:
        root: the path to the root package.
        recurse: if True, recursively search through sub-packages. Defaults to true.
        import_prefix: (optional) prefix to be added when imported. Defaults to empty.

    Raises:
        RuntimeError: if the given path isn't a valid directory or if the directory is not
            Python package with ``__init__.py``.
    """
    if not root.is_dir():
        raise RuntimeError(f"error: expected {str(root)} to be a valid directory")

    if import_prefix == "":
        import_prefix = root.stem
    else:
        import_prefix += f".{root.stem}"

    # Ensure the __init__.py exists
    init_path = root / "__init__.py"
    if not init_path.exists():
        raise RuntimeError(f"error: {str(root)} is not a Python package")

    modules: list[tuple[pathlib.Path, str]] = []
    for path in root.iterdir():
        stem = path.stem
        if path.is_dir() and recurse:
            if stem.startswith(("__", ".")):
                continue
            update_all_registries(path, True, import_prefix)

        if path.is_file() and path.suffix == ".py" and stem != "__init__":
            modules.append((path, import_prefix + f".{path.stem}"))

    import_modules: list[str] = []
    p = re.compile(r"@.+\.register\s")
    for path, tag in modules:
        with path.open("r", encoding="utf-8") as infile:
            lines = infile.readlines()
        for line in lines:
            if p.match(line):
                import_modules.append(tag)
                break

    for module in import_modules:
        importlib.import_module(module)


def safe_torch_load(
    f: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    **kwargs: typing.Any,
) -> typing.Any:
    """
    Wrap :code:`torch.load` to handle safe loading.

    This function will automatically set :code:`weights_only` to true when calling
    ``torch.load``. You are encouraged to use this function instead of the plain
    :code:`torch.load` to ensure safe loading.

    .. warning::
        :code:`weights_only` is set automatically by this function. **do not** set this
        value yourself when using this function.

    args:
        f: a file-like object (has to implement ``read()``, ``readline()``, ``tell()``,
            and ``seek()``), or a string or a ``os.pathlike`` object containing a file
            name.
        **kwargs: keyword arguments to pass to :code:`torch.load`.

    Returns:
        The result of calling :code:`torch.load`.
    """
    return torch.load(f, **kwargs, weights_only=True)
