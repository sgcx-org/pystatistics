"""
CPU survival backends.

Cox PH fitting is done directly via _cox.py (no backend abstraction needed
since it's CPU-only with no backend= parameter).

The CPU backend here is a thin wrapper for potential future use.
"""

__all__: list[str] = []
