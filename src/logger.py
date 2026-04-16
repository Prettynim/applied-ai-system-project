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

import logging    # standard library logger — used for all structured log output
import uuid       # generates random session IDs so each run is uniquely identifiable
from pathlib import Path   # object-oriented path handling; mkdir() is safer than os.makedirs

# Resolve the logs/ directory relative to this file's location — not the working directory.
# This means `python src/agent.py` and `python -m src.agent` both write to the same log file.
_LOGS_DIR = Path(__file__).parent.parent / "logs"

# Single shared format string used by both file and console handlers.
# `session_id` is injected into every record by _SessionFilter below.
_FMT = "%(asctime)s | %(levelname)-7s | session=%(session_id)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"   # human-readable timestamps without microseconds


def new_session_id() -> str:
    """
    Returns a short 8-character unique session identifier.
    Uses uuid4 (random UUID) so IDs are not guessable and don't collide across runs.
    Truncated to 8 chars for readable output — collision probability is negligible at this scale.
    """
    return uuid.uuid4().hex[:8]   # hex string, e.g. "3a8f21c0"


class _SessionFilter(logging.Filter):
    """
    Logging filter that injects a `session_id` attribute into every log record.
    Without this, the `%(session_id)s` placeholder in _FMT would raise a KeyError.
    One filter instance is created per session — each logger gets its own session_id.
    """

    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id   # stored here; injected into records in filter()

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = self.session_id   # attach session_id to every log record
        return True   # always return True — this filter never drops records


def get_logger(session_id: str) -> logging.Logger:
    """
    Returns a configured Logger for one session.

    - File handler  : logs/sessions.log  (DEBUG and above, appended)
    - Console handler: stderr            (WARNING and above only, to avoid
                                          cluttering the user-facing output)

    Calling get_logger() twice with the same session_id is safe — handlers
    are not duplicated because of the `if logger.handlers` early-return guard.
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)   # create logs/ directory if it doesn't exist

    # Use a namespaced logger name so multiple sessions don't share a logger instance
    logger_name = f"music_ai.{session_id}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger   # already configured — avoid adding duplicate handlers on repeated calls

    logger.setLevel(logging.DEBUG)    # capture everything; each handler filters independently
    logger.propagate = False          # don't pass records up to the root logger (avoids duplicate output)

    # One filter + formatter pair is shared between both handlers
    session_filter = _SessionFilter(session_id)
    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT)

    # --- File handler: writes DEBUG and above to logs/sessions.log ---
    # All sessions append to the same file — session_id in each line separates them.
    # DEBUG level captures raw profile dicts and API call details for troubleshooting.
    log_file = _LOGS_DIR / "sessions.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")   # append mode by default
    fh.setLevel(logging.DEBUG)   # capture everything in the file
    fh.setFormatter(formatter)
    fh.addFilter(session_filter)
    logger.addHandler(fh)

    # --- Console handler: writes WARNING and above to stderr ---
    # INFO/DEBUG messages are shown via the agent's own print() calls instead,
    # so users see clean, formatted output rather than duplicate timestamped log lines.
    # Only WARNING/ERROR events need to appear in the terminal — these signal problems.
    ch = logging.StreamHandler()   # writes to stderr by default
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    ch.addFilter(session_filter)
    logger.addHandler(ch)

    return logger
