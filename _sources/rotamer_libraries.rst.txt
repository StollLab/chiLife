=================
Rotamer Libraries
=================

.. _custom-rotamer-libraries:

---------------------------------
Creating Custom Rotamer Libraries
---------------------------------
A major component of chilife is the creation and sharing of custom rotamer libraries. This is to facilitate rapid
distribution of libraries for novel spin labels and encourage rapid rotamer library testing and development. Rotamer
libraries are added through the :func:`~chilife.chilife.create_library` and  :func:`~chilife.chilife.create_dlibrary`
functions:

..  code-block::

    xl.create_library('NSL', 'NSL.pdb', ...)

or by saving an existing :class:`~chilife.RotamerEnsemble` or :class:`~chilife.SpinLabel` object as a rotamer library:

..  code-block::

    mySL = xl.SpinLabel('R1M')
    # Some wizardry to optimize structures, weights, etc.
    ...
    mySL.to_rotlib('myR1')

.. note::
    chiLife :class:`~chilife.dRotamerEnsembles` and :class:`~chilife.dSpinLabels` do not yet have a ``to_rotlib``
    method.

chiLife rotamer library creation is very flexible and offers users many ways to modify their rotamer libraries.
Below we discuss several intricacies of making rotamer libraries with chiLife and general considerations that should be
taken into account when making custom rotamer libraries.

All rotamer libraries in chiLife must start with some sort of structural ensemble that approximates the side chain's
conformational landscape. This can be a very crude approximation, contain only a single or a very detailed
approximation containing several rotamers with different weights and statistical parameters describing dihedral motion
near every rotamer. As a result, there are many different ways in which a rotamer library can be developed.

The fastest and simplest way to develop a rotamer "library" is to create a PDB file of a single conformation of your
rotamer attached to a protein backbone. This can be done with several molecular modeling applications like PyMol,
Avogadro, OpenBabel, etc. Ideally the rotamer's geometry is optimized before passing it to chiLife, either with a
molecular mechanics forcefield like MMFF94, or better yet by DFT with an application like ORCA or Gaussian. For example,
we could create a completely fictional spin label that we will call "NSL":

.. image:: _static/mono_NSL.png

Once a PDB structure is generated it can be added by passing the rotlib name and pdb file to the
:func:`~chilife.chilife.create_library` function:

..  code-block::

    xl.create_library('NSL', 'NSL.pdb')

.. note::
    All labels in a pdb file passed to :func:`~chilife.chilife.create_rotlib` must have the backbone atoms named using
    standard PDB conventions and all other atoms must have unique names. Additionally all atoms must have the same
    residue number and be on the same chain.

This is the bare minimum required to generate a chilife rotlib and any :class:`~chilife.RotamerEnsemble` generated with
it will only have 1 rotamer that always models in the exact same conformation, regardless of how unfavorable that
conformation is in a given protein environment.

To allow our side chain a little flexibility we can tell chiLife what the mobile dihedrals are.


Defining Mobile Dihedrals
--------------------------

Setting Rotamer Dihedrals
-------------------------

Setting Dihedral Variances for Off Rotamer Sampling
----------------------------------------------------

Setting Rotamer Weights
-----------------------

Defining Spin-atoms and Their Weights
--------------------------------------

Using Custom Rotlibs
--------------------

--------------------------------------------------------
Differences When Creating Bifunctional Rotamer Libraries
--------------------------------------------------------
