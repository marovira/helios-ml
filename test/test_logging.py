import logging
import pathlib

import pytest

from helios.core import logging as hllog


class TestRootLogger:
    def setup_method(self) -> None:
        logger = logging.getLogger("helios")
        for h in list(logger.handlers):
            h.close()
        logger.handlers.clear()

    def teardown_method(self) -> None:
        logger = logging.getLogger("helios")
        for h in list(logger.handlers):
            h.close()
        logger.handlers.clear()

    def test_default_state(self) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        assert root.log_file is None
        assert isinstance(root.logger, logging.Logger)

    def test_setup_without_file(self) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        root.setup()
        assert root.log_file is None
        root.info("test info")
        root.warning("test warning")
        root.error("test error")

    def test_setup_with_file(self, tmp_path: pathlib.Path) -> None:
        log_file = tmp_path / "test.log"
        root = hllog.RootLogger(capture_warnings=False)
        root.setup(log_file)
        assert root.log_file == log_file
        root.info("hello from test")
        root.flush()
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "hello from test" in content

    def test_setup_appends_to_existing_file(self, tmp_path: pathlib.Path) -> None:
        log_file = tmp_path / "test.log"
        log_file.write_text("existing content\n", encoding="utf-8")
        root = hllog.RootLogger(capture_warnings=False)
        root.setup(log_file)
        root.info("new line")
        root.flush()
        content = log_file.read_text(encoding="utf-8")
        assert "existing content" in content
        assert "new line" in content

    def test_warning_and_error_logged(self, tmp_path: pathlib.Path) -> None:
        log_file = tmp_path / "messages.log"
        root = hllog.RootLogger(capture_warnings=False)
        root.setup(log_file)
        root.warning("warn msg")
        root.error("error msg")
        root.flush()
        content = log_file.read_text(encoding="utf-8")
        assert "warn msg" in content
        assert "error msg" in content

    def test_flush_close(self) -> None:
        root = hllog.RootLogger(capture_warnings=False)
        root.flush()
        root.close()


class TestDefaultLoggers:
    def setup_method(self) -> None:
        hllog.close_default_loggers()
        logger = logging.getLogger("helios")
        for h in list(logger.handlers):
            h.close()
        logger.handlers.clear()

    def teardown_method(self) -> None:
        hllog.close_default_loggers()
        logger = logging.getLogger("helios")
        for h in list(logger.handlers):
            h.close()
        logger.handlers.clear()

    def test_inactive_before_create(self) -> None:
        assert not hllog.is_root_logger_active()

    def test_get_root_logger_raises_before_create(self) -> None:
        with pytest.raises(KeyError):
            hllog.get_root_logger()

    def test_get_tensorboard_writer_none_before_create(self) -> None:
        assert hllog.get_tensorboard_writer() is None

    def test_create_and_get_root_logger(self) -> None:
        hllog.create_default_loggers(enable_tensorboard=False)
        assert hllog.is_root_logger_active()
        assert isinstance(hllog.get_root_logger(), hllog.RootLogger)

    def test_create_with_tensorboard(self) -> None:
        hllog.create_default_loggers(enable_tensorboard=True)
        writer = hllog.get_tensorboard_writer()
        assert writer is not None
        assert isinstance(writer, hllog.TensorboardWriter)

    def test_create_idempotent(self) -> None:
        hllog.create_default_loggers(enable_tensorboard=False)
        hllog.create_default_loggers(enable_tensorboard=False)
        assert hllog.is_root_logger_active()

    def test_flush_default_loggers(self) -> None:
        hllog.create_default_loggers(enable_tensorboard=False)
        hllog.flush_default_loggers()

    def test_close_default_loggers(self) -> None:
        hllog.create_default_loggers(enable_tensorboard=False)
        hllog.close_default_loggers()
        assert not hllog.is_root_logger_active()

    def test_restore_raises_without_create(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(RuntimeError):
            hllog.restore_default_loggers(log_path=tmp_path / "run.log")

    def test_setup_default_loggers(self, tmp_path: pathlib.Path) -> None:
        log_root = tmp_path / "logs"
        log_root.mkdir()
        hllog.create_default_loggers(enable_tensorboard=False)
        hllog.setup_default_loggers("myrun", log_root=log_root)
        root = hllog.get_root_logger()
        assert root.log_file is not None
        assert root.log_file.parent == log_root
        assert root.log_file.suffix == ".log"

    def test_get_default_log_name(self) -> None:
        name = hllog.get_default_log_name("myrun")
        assert name.startswith("myrun_")
        assert len(name) > len("myrun_")


class TestTensorboardWriter:
    def test_default_state(self) -> None:
        writer = hllog.TensorboardWriter()
        with pytest.raises(AssertionError):
            _ = writer.run_path

    def test_setup(self, tmp_path: pathlib.Path) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup(tmp_path)
        assert writer.run_path == tmp_path

    def test_add_scalar(self, tmp_path: pathlib.Path) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup(tmp_path)
        writer.add_scalar("loss", 0.5, 0)
        writer.flush()
        writer.close()

    def test_add_scalars(self, tmp_path: pathlib.Path) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup(tmp_path)
        writer.add_scalars("metrics", {"loss": 0.5, "acc": 0.9}, global_step=0)
        writer.flush()
        writer.close()

    def test_add_text(self, tmp_path: pathlib.Path) -> None:
        writer = hllog.TensorboardWriter()
        writer.setup(tmp_path)
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
