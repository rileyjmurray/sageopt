



Signomial nonnegativity
-----------------------

Look!

 - :func:`sageopt.relaxations.sage_sigs.sage_feasibility`, and
 - :func:`sageopt.relaxations.sage_sigs.sage_multiplier_search`.

Sageopt offers two pre-made functions for certifying nonnegativity and finding SAGE decompositions:
``sage_feasibility`` and ``sage_multiplier_search``. These functions are accessible as top-level imports
``sageopt.sage_feasibility`` and ``sageopt.sage_multiplier_search``, where they accept Signomial or Polynomial objects.
The documentation below addresses the Signomial implementations.

.. autofunction:: sageopt.relaxations.sage_sigs.sage_feasibility

.. autofunction:: sageopt.relaxations.sage_sigs.sage_multiplier_search