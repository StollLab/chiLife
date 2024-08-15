.. _rotamer_ensemble:

Side Chain Ensembles
=====================


RotamerEnsemble
---------------
The :class:`RotamerEnsemble` is the base class for (monofunctional) side chain ensemble objects. It contains all the
methods and attributes that are shared between standard canonical amino acids and specialized and non-canonical amino
acids.

.. autoclass:: chilife.RotamerEnsemble
    :members:
    :exclude-members: backbone

SpinLabel
---------

The :class:`chilife.SpinLabel` object is the primary feature of χLife. :class:`chilife.SpinLabel` inherits from the
:class:`chilife.RotamerEnsemble` object and therefore has all the same properties and methods . Additionally
:class:`chilife.SpinLabel` have several other features unique to spin labels and useful for protein and spin label
modeling.

.. autoclass:: chilife.SpinLabel
    :members:


dRotamerEnsemble
-----------------
The :class:`dRotamerEnsemble` is the base class for bifunctional side chain ensemble objects. Like
:class:`chilife.RotamerEnsemble`, it contains all the methods and attributes that are shared between all bifunctional
amino acids whether they are spin labels or other bifunctional non-canonical amino acids.

.. autoclass:: chilife.dRotamerEnsemble
    :members:

dSpinLabel
----------

.. autoclass:: chilife.dSpinLabel
    :members:
    :exclude-members: copy


IntrinsicLabel
--------------

.. autoclass:: chilife.IntrinsicLabel
    :members:

