===========================
Frequently Asked Questions
===========================

What is the difference between a :class:`chilife.RotamerEnsemble` and a :class:`chilife.SpinLabel`?
---------------------------------------------------------------------------------------------------

A :class:`chilife.RotamerEnsemble` is a more abstract class that can represent canonical amino acids as well as
non-canonical amino acids and spin labels. We will often use :class:`chilife.RotamerEnsemble`
and :class:`chilife.SpinLabel` interchangeably since chiLife is primarily an SDSL modeling application at this time. It
is important to note that  a :class:`chilife.SpinLabel` can do anything a :class:`chilife.RotamerEnsemble` can. Ideally
chilife will support additional label types in the future (e.g. FRET labels) with specialized methods and properties
that are unique to those kinds of labels.

How do I use my own PDB file?
------------------------------

chiLife uses MDAnalysis_ as a back end for protein structure and will accept :class:`MDAnalysis.Universe` and
:class:`MDAnalysis.AtomGroup` objects as proteins, thus a protein can be loaded in using MDAnalysis::

    import MDAnalysis as mda
    my_protein = mda.Universe('Path/to/my/protein.pdb')

.. _MDAnalysis: https://www.mdanalysis.org/

Many are likely already aware that MDAnalysis is generally used for trajectory analysis and can and will load in
MD trajectories and multi-state PDBs. **chiLife will ONLY create a label for the `active` frame of an MDAnalysis
trajectory**. To perform spin labeling over a trajectory you will have to manually loop through the frames and create
new :class:`~chilife.SpinLabel` objects.


What spin labels and rotamer libraries are available?
-----------------------------------------------------

Because chiLife rotamer libraries are portable, different users may have different libraries on their computers or even
in different directories. To list all the available rotamer libraries just run the
:func:`~chilife.chilife.list_available_rotlibs` function which will output a table listing the available libraries and
their locations. This function searches the chilife installation directory, any number of user defined directories
(added via the :func:`~chilife.chilife.add_rotlib_dir` function) and the current working directory. These are the exact
same places chiLife will look when create new :class:`~chilife.SpinLabel` and :class:`~chilife.RotamerEnsemble` objects
and their bifunctional analogs.

.. note::
   Additional rotamer libraries can be made by the user, provided by colleagues/collaborators, or found in the
   curated chiLife_rotlibs_ GitHub repo.

.. _chiLife_rotlibs: https://github.com/mtessmer/chiLife_rotlibs

How do I create my own rotamer libraries?
-----------------------------------------

Custom rotamer libraries are created through the :func:`~chilife.chilife.create_library` and
:func:`~chilife.chilife.create_dlibrary` functions. These functions accept multi-state PDB files that make up the
libraries structural ensemble as well as several additional arguments to specify weights, mobile dihedrals, spin atoms
and more. Check out :func:`~chilife.chilife.create_library` and :func:`~chilife.chilife.create_dlibrary` as well as the
creating custom :ref:`custom-rotamer-libraries` section.


I created my own rotamer library or got one from a collaborator now how do I use it?
------------------------------------------------------------------------------------
chiLife searches for rotamer libraries in 3 places. 1) the current working directory, 2) Any number of
:ref:`user defined folders <direct_rotlib_storage>` , 3) The default chiLife
rotamer library directory. In most cases you will just need to place the rotamer library in the directory you want to
use it in and chiLife should find it. If there is some discrepancy between the rotamer library name and the 3 letter
code used to represent the residue, then you can force the use of a specific rotamer library using the ``rotlib``
keyword argument:

.. code-block::

    xl.SpinLabel('R1A', 28, protein, rotlib='/path/to/my/specific/R1A_speciail_rotlib.npz')


Where can I find rotamer libraries for my spin label?
-----------------------------------------------------
If you are looking rotamer libraries that do not ship with chiLife by default you can find several on the curated
chiLife_rotlibs_ GitHub repo, you can :ref:`make your own <custom-rotamer-libraries>`, get them from a collaborator, or
reach out to :email:`Maxx Tessmer <mhtessmer@gmail.com>` for additional information.

.. _direct_rotlib_storage:

How do I tell chiLife where I store my personal rotamer libraries?
-------------------------------------------------------------------
You can set a user rotamer library directory or list of directories that chiLife will search before searching the
default folders. This can be done using the :func:`~chilife.chilife.add_rotlib_dir` command.

How do I emulate MMM and MTSSLWizard behavior?
-----------------------------------------------
MMM and MTSSLWizard behavior can be emulated using the :meth:`chilife.SpinLabel.from_mmm` and
:meth:`chilife.SpinLabel.from_wizard` class methods respectively. Note that even though


How do I view my rotamer/spin label ensembles?
-----------------------------------------------
You can save your ensembles (and proteins) as a pdb file using the :func:`~chilife.io.save` function and visualize
using your favorite molecular visualization software (We recommend PyMol). The function :func:`~chilife.io.save`
accepts any number of protein, :class:`~chilife.RotamerEnsemble`  and :class:`~chilife.dRotamerEnsemble` objects.

.. warning::
   If only a :class:`~chilife.SpinLabel` is given to :func:`~chilife.io.save`, chiLife will only save a
   :class:`~chilife.SpinLabel`. i.e. chilife will not save the protein that the spin label is attached to unless you
   pass the protein explicitly.

In addition to saving your ensemble, chilife will save a set of pseudo-atoms named ``NEN`` representing the
localization of the unpaired electron density if saving a :class:`~chilife.SpinLabel`. The weight of the rotamer is
mapped to the occupancy or ``q`` factor. We recommend visualizing using the following PyMol commands:

.. code-block:: none

    as surface, name NEN
    spectrum q, white_red, name NEN
    set transparency, 0.5

This will result in a surface representation with the weight of each rotamer mapped to the color intensity of the
surface.


I opened 2 PDBs with labels in PyMol but it only shows labels for one of them
------------------------------------------------------------------------------
There is a good chance that the labels have the same object name in the two files and one is being overwritten when the
second is loaded. You can alter the label name before saving, e.g.

.. code-block:: python

    omp = xl.fetch('1omp')
    anf = xl.fetch('1anf')

    # These labels will have the same name because they label the same site but they are different conformers
    SL1 = xl.Spinlabel('R1M', 41, omp)
    SL2 = xl.Spinlabel('R1M', 41, anf)

    # Add identifier 'holo' to the name of the second label before saving
    SL2.name += '_holo'

    # Saved files will now open in the same pymol window with both labels present
    xl.save(omp, SL1)
    xl.save(anf, SL2)

Why don't my label stay attached to my protein when I align it in pymol?
-------------------------------------------------------------------------------
The labels are saved as separate objects in the PDB and will not move with the protein object it is associated with.
While you can move 