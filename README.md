# chiLife
chiLife (or χLife) is a python module for modeling non-canonical amino acid side chain ensembles, primarily site 
directed spin labels (SDSLs), and using those ensembles to predict experimental results. The goal of chiLife is to provide a 
simple, flexible and interoperable python interface to protein side chain ensemble modeling, allowing for rapid 
development of custom analysis and modeling pipelines. Simplicity is facilitated by the use of `RotamerEnsemble` and 
`SpinLabel` objects with standard interfaces for all supported side chain types, side chain modeling methods and 
protein modeling methods. Flexibility is achieved by allowing users to create and use custom `RotamerEnsemble`s and 
`SpinLabel`s as well as custom side chain modeling methods. Interoperability sought by interactions with other 
Python-based molecular modeling packages. This enables the use of experimental data, like double electron-electron 
resonance (DEER), in other standalone protein modeling applications that allow user defined restraints, such as 
pyrosetta and NIH-Xplor.
 
## Getting Started
Stable distributions of chiLife can be installed using `pip`. 
```bash
pip install chiLife
```

Alternatively the development version can be installed by downloading and unpacking the GitHub repository, or using 
`git clone` followed by a standard python setuptools installation.
```bash
git clone https://github.com/mtessmer/chiLife.git
cd chiLife
pip install -e .   # Install as editable and update using `git pull origin main`
```  
***
## chiLife Module
The primary feature of chiLife is the `SpinLabel` object, which inherits from the more abstract `RotamerEnsemble` 
object. While this README primarily will refer to `SpinLabel`s, be aware that most properties and functions discussed 
are also functional on `RotamerLibrary` objects as well. `SpinLabel`s can be created and "attached" to protein models 
easily and quickly, allowing for on the fly simulation of distance distributions while modeling, or scriptable 
analysis. Notably, attaching a `SpinLabel` to a protein does not alter the protein in any way, allowing the protein 
model to retain the native amino acid.

### Simple rotamer-library based SpinLabel modeling

```python
import numpy as np
import matplotlib.pyplot as plt
import chilife as xl

# Download protein structure from PDB
MBP = xl.fetch('1omp', save=True)

# Create Spin lables
SL1 = xl.SpinLabel('R1C', site=20, chain='A', protein=MBP)
SL2 = xl.SpinLabel('R1C', site=238, chain='A', protein=MBP)

# Calculate distribution
r = np.linspace(0, 100, 256)
P = xl.distance_distribution(SL1, SL2, r=r)

# Plot distribution
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(r, P)
ax.set_yticks([])
ax.set_xlabel('Distance ($\AA$)')
for spine in ['left', 'top', 'right']:
    ax.spines[spine].set_visible(False)
plt.show()
```

![MBP L20R1 S238R1](https://github.com/StollLab/chiLife/raw/main/img/L20R1_S238R1_Pr.png)

The side chain ensembles can then be saved using a simple `save` function that accepts an arbitrary number of `
RotamerEnsemble`, `SpinLabel`, `MDAnalyisis.Universe` and `MDAnalyiss.AtomGroup` objects. Because 
`RotamerEnsemble`/`SpinLabel` objects do not mutate the underlying protein, they are saved as separate multi-state 
objects and can be visualized with applications like pymol. If you do wish to permanently alter the underlying protein
structure you can use the [`mutate`](#mutating-protein-structures) function described below.

```
# Save structure
xl.save('MBP_L20R1_S238R1.pdb', SL1, SL2, MBP)
```

![MBP L20R1 S238R1 Structure](https://github.com/StollLab/chiLife/raw/main/img/L20R1_S238R1_Structure.png)


### Mimicking MMM and MTSSLWizard
In addition to its own features, chiLife offers spin label modeling methods that mimic the popular MMM and MTSSLWizard 
modeling applications.

```python
import chilife as xl

MBP = xl.fetch('1omp')
SLmmm = xl.SpinLabel.from_mmm('R1M', site=238, protein=MBP)
SLWiz = xl.SpinLabel.from_wizard('R1M', site=238, protein=MBP,
                                 to_find=50, to_try=1000,  # Equivalent to 'quick' search, default is 'thorough'   
                                 vdw=3.4, clashes=0,  # MTSSLWizard 'tight' setting, default is 'loose' 
                                 )
```

### Off-rotamer sampling and local repacking 
One of the benefits of chiLife is the variety and customizable nature of spin label modeling methods. This includes 
methods to sample side chain conformations that deviate from canonical dihedral angles and fixed rotamer libraries 
(Off-rotamer sampling) and methods to repack a `SpinLabel` and it's neighboring amino acids, and to 

```python
import chilife as xl

MBP = xl.fetch('1omp')

# Create a SpinLabel object using the MTSSLWizard 'Accessible Volume' Approach
SL1 = xl.SpinLabel.from_wizard('R1C', site=20, chain='A', protein=MBP)

# Create a SpinLabel object by sampling off-rotamer dihedral conformations using the rotamer library as a prior 
SL2 = xl.SpinLabel('R1C', site=238, chain='A', sample=2000, protein=MBP)

# Create a SpinLabel object from a ProEPR.repack trajectory
traj, de = xl.repack(SL1, SL2, protein=MBP)
```

The repack function will perform a Markov chain Monte Carlo sampling repack of the spin labels, `SL1` and `SL2` and 
neighboring side chains, returning an `MDAnalysis.Universe` object containing all accepted structures of the MCMC 
trajectory, the energy function changes at each acceptance step and new SpinLabel objects attached to the lowest energy 
structure of the trajectory.

SpinLabel objects and neighboring side chains can be repacked using off-rotamer sampling by using the `off_rotamer=True`
option. In the event off rotamer sampling is being used for repacking, it is likely that the desired SpinLabel object is 
not the default rotamer ensembles attached to the lowest energy structure, but instead the ensemble of side chains 
created in the MCMC sampling trajectory. This can be done using the `from_trajectory` class method. 

```python
# Create a SpinLabel object from a xl.repack trajectory with off-rotamer sampling
traj, de = xl.repack(SL1, SL2, protein=MBP, off_rotamer=True) 
SL1 = xl.SpinLabel.from_trajectory(traj, site=238)
```

Off rotamer sampling can be controlled on a per dihedral basis when repacking with chiLife by passing a list of bools to 
the off_rotamer variable. For example, passing `off_rotamer = [False, False, False, True, True]` will allow for off 
rotamer sampling of only &chi;<sub>4</sub> and &chi;<sub>5</sub>.


### Mutating protein structures
Sometimes you don't want a whole rotamer ensembles, you just want a protein structure mutated at a particular site with 
the most probable spin label conformer. This can be done easily with the `mutate` function.

```python
import chilife as xl

MBP = xl.fetch('1omp')
SL = xl.SpinLabel('R1C', 238, protein=MBP)
MBP_S238R1 = xl.mutate(MBP, SL)
xl.save('MBP_S238R1.pdb', MBP_S238R1)
```

chiLife can actually mutate several sites at once, and can mutate canonical amino acids as well.

```python
SL1 = xl.SpinLabel('R1C', 20, protein=MBP)
SL2 = xl.SpinLabel('R1C', 238, protein=MBP)
L284V = xl.RotamerEnsemble('VAL', 284, protein=MBP)
```

 Mtating adjacent sites is best done with the `repack` function to avoid clashes between SpinLabels/RotamerEnsembles. 
This will return a trajectory which can be used to pick the last or lowest energy frame as your mutated protein.

```python
MBP_L284V_L20R1_S238R1, _, _ = xl.repack(SL1, SL2, L284V, protein=MBP)
```

### Adding user defined spin labels
Site directed spin labels, and other non-canonical amino acids, are constantly being developed. Additionally, rotamer 
libraries for existing labels continuously undergo incremental improvements or modification to suit particular needs, 
e.g. a rotamer library specifically for transmembrane residues. In fact chiLife iteself may be being used to develop 
new and improved, or application specific rotamer libraries. To this end chiLife makes it easy to create user defined 
spin labels and custom rotamer libraries. To create a custom rotamer library, all that is needed is (1) a pdb file of 
the spin label (2) A list of the rotatable dihedral bonds, and (3) a list of the atoms where the spin is.

```python
xl.create_library(name='TRT_1.0',
                  resname='TRT',
                  pdb='test_data/trt.pdb',
                  dihedral_atoms=[['N', 'CA', 'CB', 'SG'],
                                  ['CA', 'CB', 'SG', 'SD'],
                                  ['CB', 'SG', 'SD', 'CAD'],
                                  ['SG', 'SD', 'CAD', 'CAE'],
                                  ['SD', 'CAD', 'CAE', 'OAC']],
                  spin_atoms='CAQ')
```

This function will create a portable `TRT_1.0_rotlib.npz` file that can be called specified by the `SpinLabel` 
constructor.

```python
xl.SpinLabel('TRT', site=238, protein=MBP, rotlib='TRT_1.0', sample=5000)
```

Thus, the file can be easily shared with coworkers, collaborators or with other chiLife users via email or a 
forthcoming chiLife rotamer library repository. 

> NOTE: In the above example the `rotlib` keyword is only used for demonstration purposes. chiLife always searches the 
> current working directory for rotamer library files first. If there is a `XYZ_rotlib.npz` in the working directory 
> and you specify `xl.SpinLabel('XYZ', ...)`, chiLife will assume you want to use the `XYZ_rotlib.npz` rotamer library.

User defined labels can be constructed from a single state pdb file or a multi-state PDB file. If constructed from a 
single state pdb file a list of dihedral angles and weights can be passed via the `dihedrals` and `weigts` keyword
arguments. For each set of dihedral angles, chiLife create a rotamer and store the whole library using the specified 
name. Alternatively using a multi-state PDB file can add some additional information, such as isomeric heterogenity of 
the rotamer library, which will be maintained by chiLife.

For more information on how to use chiLife as a python module, see [examples](#examples/)
