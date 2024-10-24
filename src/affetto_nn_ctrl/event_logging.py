from __future__ import annotations

import logging
import tempfile
from pathlib import Path


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


_event_logger: logging.Logger | None = None
DEFAULT_LOG_FORMATTER = "%(asctime)s (%(module)s:%(lineno)d) [%(levelname)s]: %(message)s"


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
    if _event_logger is not None and _event_logger.name == name:
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
        pass
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


def get_event_logger() -> logging.Logger | None:
    return _event_logger


# Local Variables:
# jinx-local-words: "asctime levelname lineno noqa"
# End:
