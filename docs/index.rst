.. CF-LIBS documentation master file

CF-LIBS Documentation
=====================

**CF-LIBS** is a physics-based Python library for forward modeling and analysis of LIBS plasmas. The shipped algorithm is physics-only — see :doc:`Evolution_Framework` for the constraint spec.

Inversion Config
----------------

The classic CF-LIBS inversion uses an `analysis` section in your config file to control
line detection, selection, and solver settings. See :doc:`User_Guide` for the full schema
and :doc:`API_Reference` for CLI flags.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   Quick_Start_For_Scientists
   User_Guide
   Database_Generation
   Manifold_Generation_Guide
   Hardware_Interfaces
   Echellogram_Processing_Guide
   Deployment

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   API_Reference

.. toctree::
   :maxdepth: 1
   :caption: Developer Notes

   CF-LIBS_Codebase_Technical_Documentation
   Evolution_Framework

.. toctree::
   :maxdepth: 1
   :caption: Analysis

   REFERENCE_ANALYSIS_LIBSSA

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
