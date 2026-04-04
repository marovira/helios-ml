import helios.model as hlm
import helios.plugins as hlp
import helios.trainer as hlt


class ExceptionPlugin(hlp.Plugin):
    def __init__(self, exc_type: type[Exception] | list[type[Exception]]):
        super().__init__(f"exception_{str(exc_type)}")
        self._exc_type = exc_type

    def setup(self) -> None:
        pass

    def configure_trainer(self, t: hlt.Trainer) -> None:
        self._append_train_exceptions(self._exc_type, t)
        self._append_test_exceptions(self._exc_type, t)


class PluginModel(hlm.Model):
    def __init__(self, save_name: str):
        super().__init__(save_name)

    def setup(self, fast_init: bool = False) -> None:
        pass


@hlp.PLUGIN_REGISTRY.register
class _DummyPlugin(hlp.Plugin):
    def __init__(self) -> None:
        super().__init__("_dummy")

    def setup(self) -> None:
        pass


class TestPlugins:
    def test_registry(self, check_registry) -> None:
        check_registry(hlp.PLUGIN_REGISTRY, ["_DummyPlugin"])

    def test_create(self, check_create_function) -> None:
        check_create_function(hlp.PLUGIN_REGISTRY, hlp.create_plugin)

    def test_append_exceptions(self) -> None:
        t = hlt.Trainer()
        plugin = ExceptionPlugin(RuntimeError)
        exc_list: list[type[Exception]] = [RuntimeError]

        plugin.configure_trainer(t)
        assert t.train_exceptions == exc_list
        assert t.test_exceptions == exc_list

        exc_list.extend([ValueError, TypeError])
        plugin = ExceptionPlugin([ValueError, TypeError])
        plugin.configure_trainer(t)

        assert t.train_exceptions == exc_list
        assert t.test_exceptions == exc_list
