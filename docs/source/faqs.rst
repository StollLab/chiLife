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

    NOTE: Additional rotamer libraries can be made by the user, provided by colleagues/collaborators, or found in the
    curated chiLife_rotlibs_ GitHub repo.

.. _chiLife_rotlibs: https://github.com/mtessmer/chiLife_rotlibs

How do I create my own rotamer libraries?
-----------------------------------------

Custom rotamer libraries are created through the :func:`~chilife.chilife.create_library` and
:func:`~chilife.chilife.create_dlibrary` functions. These functions accept multi-state PDB files that make up the
libraries structural ensemble as well as several additional arguments to specify weights, mobile dihedrals, spin atoms
and more. Check out :func:`~chilife.chilife.create_library` and :func:`~chilife.chilife.create_dlibrary` as well as the
creating custom :ref:`custom-rotamer-libraries` section.


Where can I find rotamer libraries for my spin label?
-----------------------------------------------------
If you are looking rotamer libraries that do not ship with chiLife by default you can find several on the curated
chiLife_rotlibs_ GitHub repo, you can :ref:`make your own <custom-rotamer-libraries>`, get them from a collaborator, or
reach out to :email:`Maxx Tessmer <mhtessmer@gmail.com>` for additional information.


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

    NOTE: If only a :class:`~chilife.SpinLabel` is given to :func:`~chilife.io.save`, chiLife will only save a
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


