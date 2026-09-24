"""Entry point: python main.py ui.

WARDOGS supply delivery autopilot over the full map — under development.
"""

import argparse
import os
import sys


def load_config(path: str) -> dict:
    from autopilot.common.config import AppConfig

    app_cfg = AppConfig.load(path)
    return app_cfg.to_dict()


def main() -> None:
    # under pythonw.exe (no console) sys.stdout/stderr = None, and any print
    # inside the app crashes; redirect output to NUL so the app keeps running
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from autopilot import crashlog
    from autopilot.common.log import get_logger, setup_logging

    crashlog.init()  # crashes (incl. native cv2) are written to output/crash.log
    setup_logging()  # standard logs written to output/autopilot.log
    logger = get_logger("main")
    logger.info("Starting WARDOGS autopilot...")

    ap = argparse.ArgumentParser(
        prog="wardogs-autopilot",
        description="WARDOGS supply delivery autopilot over the WARDOGS minimap.",
    )
    ap.add_argument("--config", default="config.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ui = sub.add_parser("ui", help="Studio: capture zone + live map + routes (GUI)")

    args = ap.parse_args()
    cfg = load_config(args.config)

    try:
        if args.cmd == "ui":
            from autopilot import ui

            code = ui.main(cfg)
            if code not in (0, None):
                sys.exit(code)
    except (SystemExit, KeyboardInterrupt) as exc:
        if isinstance(exc, SystemExit) and exc.code not in (0, None):
            from autopilot import crashlog as cl

            cl.write(*sys.exc_info())
            raise
    except Exception:
        from autopilot import crashlog as cl

        cl.write(*sys.exc_info())
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            _app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "WARDOGS Studio", "Application error.\nDetails: output/crash.log")
        except Exception:
            try:
                from tkinter import messagebox

                messagebox.showerror("WARDOGS Studio", "Application error.\nDetails: output/crash.log")
            except Exception:
                pass


if __name__ == "__main__":
    main()
