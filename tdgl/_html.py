"""Notebook HTML display without an eager IPython dependency."""

import sys


class HTML:
    """Minimal object implementing Jupyter's rich HTML display protocol."""

    def __init__(self, data: str):
        self.data = data

    def _repr_html_(self) -> str:
        return self.data


def make_html(data: str):
    """Return IPython's HTML wrapper when it is safe, otherwise a fallback."""
    # Anaconda's Python 3.13 rlcompleter currently segfaults while IPython
    # imports pdb. The display protocol itself needs no IPython runtime.
    if sys.version_info < (3, 13):
        try:
            from IPython.display import HTML as IPythonHTML

            return IPythonHTML(data)
        except ImportError:
            pass
    return HTML(data)
