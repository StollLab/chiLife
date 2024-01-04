Molecular Systems
=================

Under construction. The ``chiLife.MolSys`` module is not yet officially supported.


The MolSys object
-----------------

The chiLife Molecular System, or ``MolSys`` and related objects, are how chilife represents molecular systems including
proteins, nucleic acids, small molecules, course grained systems and more. ``MolSys`` objects are generally created
from PDB files, MDAnalysis objects or existing chilife molecular systems. The design of the chiLife molecular system
objects are heavily influenced by `MDAnalysis <https://userguide.mdanalysis.org/stable/index.html>`_ and shares many
attribute and method names to facilitate a compatible interface.



.. automodule:: chilife.MolSys
    :members: MolecularSystemBase,
              MolSys,
              Trajectory,
              AtomSelection,
              ResidueSelection,
              SegmentSelection,
              Atom,
              Residue,
              Segment


The MolSysIC object
-------------------

While the :class:`~MolSys` object contains cartesian coordinate and protein information, the  :class:`~MolSysIC`
object contains information on internal coordinates.

.. automodule:: chilife.MolSysIC
    :members:

