from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

DEFAULT_LOG_FORMATTER = "%(asctime)s (%(module)s:%(lineno)d) [%(levelname)s]: %(message)s"


class FakeLogger:
    _console_output: bool
    _logging_level: int
    _logging_level_map: ClassVar[dict[str, int]] = {
        "NOTSET": 0,
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }

    def __init__(self, *, disable_console_output: bool = False) -> None:
        self._disable_console_output = disable_console_output
        self._logging_level = logging.WARNING

    def _format(self, msg: str, *args: object) -> str:
        formatted_msg = msg % args
        if not formatted_msg.endswith("\n"):
            formatted_msg += "\n"
        return formatted_msg

    def log(self, level: int, msg: str, *args: object, stacklevel: int = 1) -> None:
        _ = stacklevel
        match level:
            case _ if level >= logging.CRITICAL:
                self.critical(msg, args, stacklevel=stacklevel)
            case _ if level >= logging.ERROR:
                self.error(msg, args, stacklevel=stacklevel)
            case _ if level >= logging.WARNING:
                self.warning(msg, args, stacklevel=stacklevel)
            case _ if level >= logging.INFO:
                self.info(msg, args, stacklevel=stacklevel)
            case _:
                self.debug(msg, args, stacklevel=stacklevel)

    def debug(self, msg: str, *args: object, stacklevel: int = 1) -> None:
        _ = stacklevel
        if not self._disable_console_output and self._logging_level <= logging.DEBUG:
            sys.stderr.write(self._format(msg, *args))

    def info(self, msg: str, *args: object, stacklevel: int = 1) -> None:
        _ = stacklevel
        if not self._disable_console_output and self._logging_level <= logging.INFO:
            sys.stderr.write(self._format(msg, *args))

    def warning(self, msg: str, *args: object, stacklevel: int = 1) -> None:
        _ = stacklevel
        if not self._disable_console_output and self._logging_level <= logging.WARNING:
            sys.stderr.write(self._format(msg, *args))

    def error(self, msg: str, *args: object, stacklevel: int = 1) -> None:
        _ = stacklevel
        if not self._disable_console_output and self._logging_level <= logging.ERROR:
            sys.stderr.write(self._format(msg, *args))

    def critical(self, msg: str, *args: object, stacklevel: int = 1) -> None:
        _ = stacklevel
        if not self._disable_console_output and self._logging_level <= logging.CRITICAL:
            sys.stderr.write(self._format(msg, *args))

    def suppress_console_output(self) -> None:
        self._disable_console_output = False

    def enable_console_output(self) -> None:
        self._disable_console_output = True

    def set_logging_level(self, logging_level: int | str) -> None:
        if isinstance(logging_level, int):
            self._logging_level = logging_level
        else:
            self._logging_level = self._logging_level_map[logging_level]

    def setLevel(self, level: int | str) -> None:  # noqa: N802
        self.set_logging_level(level)

    @property
    def logging_level(self) -> int:
        return self._logging_level


def is_running_in_pytest() -> bool:
    return "pytest" in sys.modules


_event_logger: logging.Logger | FakeLogger = FakeLogger(disable_console_output=is_running_in_pytest())


def _get_default_event_log_filename(
    argv: list[str],
    output_dir: str | Path | None,
    given_log_filename: str | Path | None,
) -> Path:
    event_log_filename: Path
    if given_log_filename is None:
        if output_dir is not None:
            event_log_filename = (Path(output_dir) / Path(argv[0]).name).with_suffix(".log")
        else:
            event_log_filename = (Path(tempfile.gettempdir()) / Path(argv[0]).name).with_suffix(".log")
    else:
        event_log_filename = Path(given_log_filename)

    return event_log_filename


def start_event_logging(
    argv: list[str],
    output_dir: str | Path | None = None,
    log_filename: str | Path | None = None,
    name: str | None = None,
    logging_level: int | str = logging.WARNING,
    logging_level_file: int | str = logging.DEBUG,
    fmt: str | None = None,
) -> logging.Logger:
    if fmt is None:
        fmt = DEFAULT_LOG_FORMATTER
    if name is None:
        name = __name__

    global _event_logger  # noqa: PLW0603
    if isinstance(_event_logger, logging.Logger) and _event_logger.name == name:
        # Event logging has been started.
        return _event_logger

    # Create a new logger instance.
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter(fmt)

    # Setup a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Setup a file handler
    event_log_filename = _get_default_event_log_filename(argv, output_dir, log_filename)
    try:
        fh = logging.FileHandler(event_log_filename)
    except FileNotFoundError:
        # Maybe, running in dry-run mode...
        sys.stderr.write(f"Unable to save log file (if running in dry-run mode, ignore this): {event_log_filename}\n")
    else:
        fh.setLevel(logging_level_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Always log command arguments.
    logger.debug("Start event logging")
    logger.debug("Logger name: %s", logger.name)
    logger.debug("Log filename: %s", event_log_filename)
    cmd = "python " + " ".join(argv)
    logger.debug("Command: %s", cmd)

    _event_logger = logger
    return _event_logger


def event_logger() -> logging.Logger | FakeLogger:
    return _event_logger


def get_logging_level_from_verbose_count(verbose_count: int) -> str:
    match verbose_count:
        case 0:
            return "WARNING"
        case 1:
            return "INFO"
        case _:
            return "DEBUG"


def start_logging(
    argv: list[str],
    output_dir: Path,
    name: str,
    verbose_count: int,
    *,
    dry_run: bool = False,
) -> logging.Logger:
    logging_level = get_logging_level_from_verbose_count(verbose_count)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    return start_event_logging(argv, output_dir, name=name, logging_level=logging_level)


# Local Variables:
# jinx-local-words: "asctime levelname lineno noqa pytest"
# End:
