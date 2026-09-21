import os
import sys

# Ensure project root and src are on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Ensure Tcl/Tk libraries are located reliably on Windows
for base in [r"C:\Python313\tcl", os.path.join(sys.base_prefix, "tcl")]:
    tcl_dir = os.path.join(base, "tcl8.6")
    tk_dir = os.path.join(base, "tk8.6")
    if os.path.isdir(tcl_dir) and "TCL_LIBRARY" not in os.environ:
        os.environ["TCL_LIBRARY"] = tcl_dir
    if os.path.isdir(tk_dir) and "TK_LIBRARY" not in os.environ:
        os.environ["TK_LIBRARY"] = tk_dir
