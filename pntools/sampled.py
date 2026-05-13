"""Pickle-compat shim — kept after the 2026-05-13 import retirement pass.

Active code no longer imports from here; the pntools.sampled namespace
exists solely so existing on-disk pickles (~292 in C:/data/_cache as of
the retirement) that reference `pntools.sampled.Data` as the class
module continue to load. Full deletion is deferred (see
pn-specs/plans/20260513_pntools_sampled_gui_retirement.md).
"""
from pysampled import *
