"""
Session logging for the Music Recommender AI system.

Writes structured log entries to logs/sessions.log for every session.

Logged events:
  INFO    — session start/end, steps completed, Claude API calls
  WARNING — guardrail issues (WARNING/ERROR severity), low confidence extractions
  ERROR   — API failures, file-not-found, unexpected exceptions
  DEBUG   — raw profile dicts, recommendation scores, full critique text

Usage:
    from logger import get_logger, new_session_id

    session_id = new_session_id()
    log = get_logger(session_id)
    log.info("Session started")
    log.warning("Guardrail fired: GENRE_NOT_IN_CATALOG")
    log.error("Claude API call failed: ...")
"""

import logging
import uuid
from pathlib import Path

_LOGS_DIR = Path(__file__).parent.parent / "logs"

# Single shared format for both file and console handlers
_FMT = "%(asctime)s | %(levelname)-7s | session=%(session_id)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def new_session_id() -> str:
    """Returns a short 8-character unique session identifier."""
    return uuid.uuid4().hex[:8]


class _SessionFilter(logging.Filter):
    """Injects session_id into every log record for a logger."""

    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = self.session_id
        return True


def get_logger(session_id: str) -> logging.Logger:
    """
    Returns a configured Logger for one session.

    - File handler  : logs/sessions.log  (DEBUG and above, appended)
    - Console handler: stderr            (WARNING and above only, to avoid
                                          cluttering the user-facing output)

    Calling get_logger() twice with the same session_id is safe — handlers
    are not duplicated.
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger_name = f"music_ai.{session_id}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't bubble up to root logger

    session_filter = _SessionFilter(session_id)
    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT)

    # --- File handler: captures everything including DEBUG ---
    log_file = _LOGS_DIR / "sessions.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    fh.addFilter(session_filter)
    logger.addHandler(fh)

    # --- Console handler: WARNING and above only ---
    # INFO/DEBUG messages are shown via the agent's own print() calls instead,
    # so users see clean output rather than duplicate timestamped lines.
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    ch.addFilter(session_filter)
    logger.addHandler(ch)

    return logger
