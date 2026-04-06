import logging as stdlib_logging
import pathlib
import typing

import pytest

from helios.core import loggers as hllog
from helios.core.loggers import LoggerType
from helios.core.loggers.base import Logger


def _clear_helios_handlers() -> None:
    logger = stdlib_logging.getLogger("helios")
    for h in list(logger.handlers):
        h.close()
    logger.handlers.clear()


class TestLoggerBase:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            Logger()  # type: ignore[abstract]

    def test_logger_type_root_value(self) -> None:
        assert LoggerType.ROOT.value == "root"

    def test_logger_type_tensorboard_value(self) -> None:
        assert LoggerType.TENSORBOARD.value == "tensorboard"

    def test_logger_type_members(self) -> None:
        members = {t.name for t in LoggerType}
        assert "ROOT" in members
        assert "TENSORBOARD" in members


class TestRootLogger:
    def setup_method(self) -> None:
        _clear_helios_handlers()

    def teardown_method(self) -> None:
        _clear_helios_handlers()

    def test_is_logger_subclass(self) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        assert isinstance(root, Logger)

    def test_default_state(self) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        assert root.log_file is None
        assert isinstance(root.logger, stdlib_logging.Logger)

    def test_setup_without_log_root(self) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        root.setup("myrun", None, is_resume=False)
        assert root.log_file is None

    def test_setup_creates_log_file(self, tmp_path: pathlib.Path) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        root.setup("myrun", tmp_path, is_resume=False)
        assert root.log_file is not None
        assert root.log_file.parent == tmp_path
        assert root.log_file.suffix == ".log"

    def test_setup_writes_to_file(self, tmp_path: pathlib.Path) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        root.setup("myrun", tmp_path, is_resume=False)
        root.info("hello from test")
        root.flush()
        assert root.log_file is not None
        content = root.log_file.read_text(encoding="utf-8")
        assert "hello from test" in content

    def test_setup_appends_to_existing_file(self, tmp_path: pathlib.Path) -> None:
        log_file = tmp_path / "existing.log"
        log_file.write_text("prior content\n", encoding="utf-8")

        root = hllog.RootLogger(capture_warnings=False)
        # Manually inject the saved path so setup() treats it as a resume.
        root.load_state_dict({"log_file": log_file})
        root.setup("myrun", tmp_path, is_resume=True)
        root.info("new entry")
        root.flush()

        content = log_file.read_text(encoding="utf-8")
        assert "prior content" in content
        assert "new entry" in content

    def test_resume_reuses_saved_log_file(self, tmp_path: pathlib.Path) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        root.setup("myrun", tmp_path, is_resume=False)
        first_log_file = root.log_file

        state = root.state_dict()
        _clear_helios_handlers()

        root2 = hllog.RootLogger(capture_warnings=False)
        root2.load_state_dict(state)
        root2.setup("myrun", tmp_path, is_resume=True)
        assert root2.log_file == first_log_file

    def test_resume_without_saved_state_generates_new_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        root.setup("myrun", tmp_path, is_resume=True)
        assert root.log_file is not None

    def test_state_dict_contains_log_file(self, tmp_path: pathlib.Path) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        root.setup("myrun", tmp_path, is_resume=False)
        state = root.state_dict()
        assert "log_file" in state
        assert state["log_file"] == root.log_file

    def test_state_dict_no_file(self) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        state = root.state_dict()
        assert state["log_file"] is None

    def test_warning_and_error_written_to_file(self, tmp_path: pathlib.Path) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        root.setup("myrun", tmp_path, is_resume=False)
        root.warning("warn msg")
        root.error("error msg")
        root.flush()
        assert root.log_file is not None
        content = root.log_file.read_text(encoding="utf-8")
        assert "warn msg" in content
        assert "error msg" in content

    def test_flush_and_close_without_setup(self) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        root.flush()
        root.close()


class TestTensorboardWriter:
    def test_is_logger_subclass(self) -> None:
        writer = hllog.TensorboardWriter()
        assert isinstance(writer, Logger)

    def test_default_state(self) -> None:
        writer = hllog.TensorboardWriter()
        with pytest.raises(AssertionError):
            _ = writer.run_path

    def test_setup_without_log_root(self) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup("myrun", None, is_resume=False)
        with pytest.raises(AssertionError):
            _ = writer.run_path

    def test_setup_creates_run_path(self, tmp_path: pathlib.Path) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup("myrun", tmp_path, is_resume=False)
        assert writer.run_path is not None
        assert writer.run_path.parent == tmp_path / "tensorboard"
        writer.close()

    def test_resume_reuses_saved_run_path(self, tmp_path: pathlib.Path) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup("myrun", tmp_path, is_resume=False)
        first_run_path = writer.run_path
        state = writer.state_dict()
        writer.close()

        writer2 = hllog.TensorboardWriter()
        writer2.load_state_dict(state)
        writer2.setup("myrun", tmp_path, is_resume=True)
        assert writer2.run_path == first_run_path
        writer2.close()

    def test_resume_without_saved_state_generates_new_path(
        self, tmp_path: pathlib.Path
    ) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup("myrun", tmp_path, is_resume=True)
        assert writer.run_path is not None
        writer.close()

    def test_state_dict_contains_run_path(self, tmp_path: pathlib.Path) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup("myrun", tmp_path, is_resume=False)
        state = writer.state_dict()
        assert "run_path" in state
        assert state["run_path"] == writer.run_path
        writer.close()

    def test_state_dict_no_path(self) -> None:
        writer = hllog.TensorboardWriter()
        state = writer.state_dict()
        assert state["run_path"] is None

    def test_add_scalar(self, tmp_path: pathlib.Path) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup("myrun", tmp_path, is_resume=False)
        writer.add_scalar("loss", 0.5, 0)
        writer.flush()
        writer.close()

    def test_add_scalars(self, tmp_path: pathlib.Path) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup("myrun", tmp_path, is_resume=False)
        writer.add_scalars("metrics", {"loss": 0.5, "acc": 0.9}, global_step=0)
        writer.flush()
        writer.close()

    def test_add_text(self, tmp_path: pathlib.Path) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup("myrun", tmp_path, is_resume=False)
        writer.add_text("config", "lr=0.001", global_step=0)
        writer.flush()
        writer.close()

    def test_methods_noop_without_setup(self) -> None:
        writer = hllog.TensorboardWriter()
        writer.add_scalar("loss", 0.5, 0)
        writer.add_scalars("metrics", {"loss": 0.5}, global_step=0)
        writer.add_text("config", "lr=0.001")
        writer.flush()
        writer.close()


class TestLogging:
    def setup_method(self) -> None:
        hllog.close_loggers()
        _clear_helios_handlers()

    def teardown_method(self) -> None:
        hllog.close_loggers()
        _clear_helios_handlers()

    def test_inactive_before_create(self) -> None:
        assert not hllog.is_root_logger_active()

    def test_get_logger_raises_before_create(self) -> None:
        with pytest.raises(KeyError):
            hllog.get_logger(LoggerType.ROOT)

    def test_create_always_adds_root_logger(self) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        assert hllog.is_root_logger_active()
        root = hllog.get_logger(LoggerType.ROOT)
        assert isinstance(root, hllog.RootLogger)

    def test_create_with_tensorboard(self) -> None:
        hllog.create_loggers(enable_tensorboard=True)
        writer = hllog.get_logger(LoggerType.TENSORBOARD)
        assert isinstance(writer, hllog.TensorboardWriter)

    def test_create_without_tensorboard_raises_on_get(self) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        with pytest.raises(KeyError):
            hllog.get_logger(LoggerType.TENSORBOARD)

    def test_create_idempotent(self) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        hllog.create_loggers(enable_tensorboard=False)
        assert hllog.is_root_logger_active()

    def test_setup_creates_log_file(self, tmp_path: pathlib.Path) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        hllog.setup_loggers("myrun", tmp_path)
        root = typing.cast(hllog.RootLogger, hllog.get_logger(LoggerType.ROOT))
        assert root.log_file is not None
        assert root.log_file.parent == tmp_path

    def test_setup_with_no_log_root(self) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        hllog.setup_loggers("myrun", None)
        root = typing.cast(hllog.RootLogger, hllog.get_logger(LoggerType.ROOT))
        assert root.log_file is None

    def test_restore_fresh_creates_new_file(self, tmp_path: pathlib.Path) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        hllog.restore_loggers("myrun", tmp_path)
        root = typing.cast(hllog.RootLogger, hllog.get_logger(LoggerType.ROOT))
        assert root.log_file is not None

    def test_restore_with_state_reuses_log_file(self, tmp_path: pathlib.Path) -> None:
        # First run.
        hllog.create_loggers(enable_tensorboard=False)
        hllog.setup_loggers("myrun", tmp_path)
        state = hllog.get_logger_state_dicts()
        saved_log_file = typing.cast(
            hllog.RootLogger, hllog.get_logger(LoggerType.ROOT)
        ).log_file
        hllog.close_loggers()
        _clear_helios_handlers()

        # Resume run.
        hllog.create_loggers(enable_tensorboard=False)
        hllog.restore_loggers("myrun", tmp_path, state)
        root = typing.cast(hllog.RootLogger, hllog.get_logger(LoggerType.ROOT))
        assert root.log_file == saved_log_file

    def test_restore_with_none_state_is_fresh(self, tmp_path: pathlib.Path) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        hllog.restore_loggers("myrun", tmp_path, None)
        root = typing.cast(hllog.RootLogger, hllog.get_logger(LoggerType.ROOT))
        assert root.log_file is not None

    def test_restore_logger_absent_from_state_gets_fresh_setup(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Create both root and tensorboard, but save state with only root.
        hllog.create_loggers(enable_tensorboard=True)
        hllog.setup_loggers("myrun", tmp_path)
        root_only_state = {"root": hllog.get_logger_state_dicts()["root"]}
        hllog.close_loggers()
        _clear_helios_handlers()

        hllog.create_loggers(enable_tensorboard=True)
        hllog.restore_loggers("myrun", tmp_path, root_only_state)
        # Tensorboard was absent from state, so it gets a fresh setup.
        writer = typing.cast(
            hllog.TensorboardWriter, hllog.get_logger(LoggerType.TENSORBOARD)
        )
        assert writer.run_path is not None
        writer.close()

    def test_get_logger_state_dicts_contains_root(self, tmp_path: pathlib.Path) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        hllog.setup_loggers("myrun", tmp_path)
        state = hllog.get_logger_state_dicts()
        assert "root" in state
        assert "log_file" in state["root"]

    def test_get_logger_state_dicts_contains_tensorboard(
        self, tmp_path: pathlib.Path
    ) -> None:
        hllog.create_loggers(enable_tensorboard=True)
        hllog.setup_loggers("myrun", tmp_path)
        state = hllog.get_logger_state_dicts()
        assert "tensorboard" in state
        assert "run_path" in state["tensorboard"]
        typing.cast(
            hllog.TensorboardWriter, hllog.get_logger(LoggerType.TENSORBOARD)
        ).close()

    def test_flush_loggers(self) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        hllog.flush_loggers()

    def test_close_loggers_clears_table(self) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        hllog.close_loggers()
        assert not hllog.is_root_logger_active()

    def test_get_default_log_name(self) -> None:
        name = hllog.get_default_log_name("myrun")
        assert name.startswith("myrun_")
        assert len(name) > len("myrun_")

    def test_get_root_logger_compat(self) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        root = hllog.get_root_logger()
        assert isinstance(root, hllog.RootLogger)

    def test_get_root_logger_raises_when_inactive(self) -> None:
        with pytest.raises(KeyError):
            hllog.get_root_logger()

    def test_get_tensorboard_writer_returns_none_when_disabled(self) -> None:
        hllog.create_loggers(enable_tensorboard=False)
        assert hllog.get_tensorboard_writer() is None

    def test_get_tensorboard_writer_returns_instance_when_enabled(self) -> None:
        hllog.create_loggers(enable_tensorboard=True)
        writer = hllog.get_tensorboard_writer()
        assert isinstance(writer, hllog.TensorboardWriter)
