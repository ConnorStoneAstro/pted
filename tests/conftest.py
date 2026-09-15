"""Force a non-interactive matplotlib backend for the whole test suite.

Several tests call the plotting helpers, and importing ``pyplot`` picks a
backend the first time. Left alone matplotlib prefers an interactive one
wherever it can find a GUI toolkit, so on a machine with tkinter installed but
no usable display -- a Windows CI runner, a headless workstation, an ssh
session -- ``pit_plot`` dies inside Tk rather than writing its file:

    _tkinter.TclError: Can't find a usable init.tcl

Agg writes files and touches no GUI, which is all these tests want. Set here
rather than in the library, which has no business overriding a backend the
caller chose deliberately.
"""

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:  # the suite covers the no-matplotlib path too
    pass
