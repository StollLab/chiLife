Scoring
=============

This module consists of several built in scoring functions as well as helper functions to setup scoring. In
addition to the built in scoring functions, users can also define their own scoring functions. The only
requirements are that the scoring function accepts a RotamerEnsemble or SpinLabel object and outputs an energy score
for each rotamer in the ensemble. The energy score should be in kcal/mol.

.. automodule:: chilife.scoring
    :members:
