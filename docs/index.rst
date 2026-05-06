.. CF-LIBS documentation master file

CF-LIBS Documentation
=====================

**CF-LIBS** is a physics-based Python library for forward modeling and
inversion of laser-induced breakdown spectroscopy plasmas. The shipped
algorithm is physics-only — see :doc:`development/Evolution_Framework`
for the constraint specification.

Documentation is organized into four areas:

* **User guides** — how to analyze real spectra and generate synthetic
  ones.
* **Physics reference** — equations, assumptions, and the inversion
  algorithm in detail.
* **Reference** — API and codebase architecture.
* **Development** — deployment, evolution framework, and other internal
  notes.

For new users analyzing measured LIBS spectra, start with
:doc:`user/Quick_Start_Real_Data`.

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   user/Quick_Start_Real_Data
   user/Quick_Start_Synthetic
   user/User_Guide
   user/Manifold_Generation_Guide
   user/Hardware_Interfaces
   user/Echellogram_Processing_Guide

.. toctree::
   :maxdepth: 2
   :caption: Physics

   physics/README
   physics/Equations
   physics/Assumptions_And_Validity
   physics/Inversion_Algorithm

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/API_Reference
   reference/Codebase_Architecture
   reference/Database_Generation

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/Evolution_Framework
   development/Deployment
   development/CODEEVOLVE_WAVE2_PLAN
   development/REFERENCE_ANALYSIS_LIBSSA

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
