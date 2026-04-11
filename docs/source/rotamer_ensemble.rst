.. _rotamer_ensemble:

Molecular Ensembles
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

The :class:`chilife.SpinLabel` inherits from the :class:`chilife.RotamerEnsemble` object and therefore has all the same
properties and methods. Additionally :class:`chilife.SpinLabel` have several other features unique to spin labels and
useful for protein and spin label modeling.

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
The :class:`dSpinLabel` class is the radicalized extension of the :class:`dRotamerEnsemble`. It is used to model
bifunctional spinl labels like RX and di-histidine copper capped with NTA.

.. autoclass:: chilife.dSpinLabel
    :members:
    :exclude-members: copy


LigandEnsemble
--------------
The :class:`LigandEnsemble` is the base class for ligands. Ligands, being free-floating structures rather than branches
off a protein or nucleic acid chain, must be treated very differently than other ensemble objects. Nonetheless, the
:class:`LigandEnsemble` API is designed to be used in a very similar manner to the :class:`RotamerEnsemble` and
:class:`dRotamerEnsemble` class families.

.. autoclass:: chilife.LigandEnsemble
    :members:


SpinLigand
----------
The :class:`SpinEnsemble` class extends :class:`LigandEnsemble` allowing for free radicals and can be used like a
:class:`SpinLabel` to get distance distributions with other :class:`SpinLabel` like objects and visualize spin density.

.. autoclass:: chilife.SpinLigand
    :members:


IntrinsicLabel
--------------
The :class:`IntrinsicLabel` class is designed to allow users to specify free radicals that are already present in a
molecular system or protein structure such as metals in hemes and porphyrins, organic radicals like the tryptophan
cation radical.

.. autoclass:: chilife.IntrinsicLabel
    :members:
