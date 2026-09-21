"""Crash (traceback) logging to output/crash.log.

Runs under pythonw.exe: there is no console, all stderr output goes to NUL —
any app error silently disappears and looks like "the app crashed". This
module intercepts:
  * sys.excepthook            — exceptions in the main thread / Tk;
  * threading.excepthook      — exceptions in threads (locator, navigator);
  * faulthandler              — native cv2/Tk crashes (segfault, no Python);
  * explicit log()/write()    — called from code that catches an error but
                                still needs to record that it happened
                                (locator thread).
"""

import faulthandler
import os
import sys
import threading
import time
import traceback

_LOG = None
_LOCK = threading.Lock()


def init() -> object:
    """Enable all hooks. Safe to call multiple times."""
    global _LOG
    if _LOG is not None:
        return _LOG
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", "crash.log")
    _LOG = open(path, "a", encoding="utf-8")
    _LOG.write(
        "\n===== startup %s (python %s) =====\n"
        % (time.strftime("%Y-%m-%d %H:%M:%S"), sys.version.split()[0])
    )
    _LOG.flush()
    sys.excepthook = _excepthook
    try:
        threading.excepthook = _thread_hook
    except AttributeError:  # not in old pythons
        pass
    try:
        faulthandler.enable(file=_LOG, all_threads=True)
    except Exception:  # noqa: BLE001
        pass
    return _LOG


def _excepthook(tp, val, tb) -> None:
    write(tp, val, tb)


def _thread_hook(args) -> None:
    write(args.exc_type, args.exc_value, args.exc_traceback)


def write(tp=None, val=None, tb=None) -> None:
    """Writes one exception (or message) to crash.log."""
    with _LOCK:
        try:
            if _LOG is None:
                return
            _LOG.write(
                "\n--- %s  %s ---\n" % (time.strftime("%H:%M:%S"), threading.current_thread().name)
            )
            if tb is not None:
                traceback.print_exception(tp, val, tb, file=_LOG)
            elif val is not None:
                _LOG.write("%s: %s\n" % (tp.__name__ if tp else "error", val))
            else:
                _LOG.write("%s\n" % (val if val else "(no message)"))
            _LOG.flush()
        except Exception:  # noqa: BLE001 — nothing matters more than the log
            pass


def log(tag: str, exc: BaseException | None = None) -> None:
    """Short record from code: tag + exception type/text (if any)."""
    if exc is None:
        write(tag)
    else:
        write(type(exc), exc, exc.__traceback__)
