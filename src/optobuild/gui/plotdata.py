"""Plot-ready curves for the GUI: re-exports ``optobuild.reporting.figures``.

The curve builders live in the reporting layer so that reports and the GUI draw
the same data; this module keeps the GUI's import path stable.
"""

from optobuild.reporting.figures import *  # noqa: F403
from optobuild.reporting.figures import __all__  # noqa: F401
