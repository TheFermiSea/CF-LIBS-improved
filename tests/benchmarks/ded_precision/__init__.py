"""Synthetic DED composition-series precision benchmark (DED-PLAN section 4).

Forward-models AM alloys (Ti-6Al-4V, Inconel 625, 316L) across a composition
series, adds realistic DED LIBS noise, extracts intensities at known line
positions (bypassing peak detection AND element ID), runs the constrained-
element solver, and measures ABSOLUTE recovered wt% vs the known input plus
precision/ratio/Delta-sensitivity. No external data required.
"""
