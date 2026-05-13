"""Pickle-compat shim — kept after the 2026-05-13 import retirement pass.

Active code no longer imports from here; the pntools.gui namespace
exists solely as a precaution for any on-disk pickles that may
reference it (none found under C:/data/_cache at retirement time, but
NAS / project-dir / external caches were not exhaustively audited).
Full deletion is deferred (see
pn-specs/plans/20260513_pntools_sampled_gui_retirement.md).
"""
from datanavigator import *
