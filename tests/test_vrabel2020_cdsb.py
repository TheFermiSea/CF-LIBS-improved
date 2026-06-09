"""Vrabel-2020 CD-SB tuning tests.

The two tests in this module exercised the legacy ``CDSBPlotter`` (its default
tuning constants and the ``_estimate_initial_tau`` resonance-boost heuristic).
``CDSBPlotter`` was removed (audit defects #14b/#14c/#13c) and replaced by the
production :class:`cflibs.inversion.physics.self_absorption.SelfAbsorptionCorrector`,
whose optical-depth model is covered by ``tests/test_self_absorption.py``. The
``CDSBPlotter``-specific tests were removed with the class; nothing in the
Vrabel-2020 tuning surface survives to test here.
"""
