"""
Stosh: Object-oriented Python interface to Stan

A simple Python interface to Stan's sampling capabilities using the stan::run C++ API.
"""

from .stosh import compile, CompiledModel, StoshError

__version__ = "0.1.0"
__all__ = ["compile", "CompiledModel", "StoshError"]
