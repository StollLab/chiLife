# ProEPR
ProEPR is a python module for the simulation and analysis of <u>Pro</u>tein <u>EPR</u> experiments. The primary purpose  
of ProEPR is to perform *in silico* Site Directed Spin Labeling (SDSL) to simulate experimental results. Rapid, 
accurate and scriptable experimental simulations are necessary for the conversion of experimental data into high quality 
protein models. ProEPR aims to achieve this by combining standard side chain modeling methods with commonly used spin 
labels. Furthermore, ProEPR offers a scriptable environment allowing users to develop novel protocols unique to their 
protein modeling tasks.
 
## Getting Started
ProEPR can be installed by downloading and unpacking the GitHub repository, or using `git clone` followed by a standard 
python setuptools installation.
```bash
git clone https://github.com/mtessmer/ProEPR
cd ProEPR
python setup.py install
```  
>NOTE: there is currently a conflict between `numba` (0.55.1) and `mdanalysis` (2.0.0) that may cause some issues. 
> `mdanalysis` 2.0.0 on pypi appears to be built with `numpy` 1.22 but `numba` does not yet support `numpy>=1.21`. To 
> overcome these conflicts `mdanalysis` can be built on installation with a fixed `numpy==1.21` version. 
> ```bash
> pip install numpy==1.21.5
> pip install mdanalysis --no-binary mdanalysis
> ```
> before running the ProEPR setup script. Note that this may requires system specific build tools.

***
## ProEPR Module
ProEPR is most powerful when used as an API for your data analysis and protein modeling pipeline. The primary feature of 
ProEPR is the `SpinLabel` object. `SpinLabel`s can be created and attached to protein models easily and quickly, allowing for 
on the fly simulation of distance distributions while modeling. Because ProEPR is written in python, in can be easily 
integrated into many python based modeling workflows. Creating a SpinLabel object is easy:

### Simple rotamer-library based SpinLabel modeling
```python
import numpy as np
import matplotlib.pyplot as plt
import ProEPR

# Download protein structure from PDB
MBP = ProEPR.fetch('1omp', save=True)

# Create Spin lables
SL1 = ProEPR.SpinLabel('R1C', site=20, chain='A', protein=MBP)
SL2 = ProEPR.SpinLabel('R1C', site=238, chain='A', protein=MBP)

# Calculate distribution
r = np.linspace(0, 100, 256)
P = ProEPR.get_dd(SL1, SL2, r=r)

# Plot distribution
fig, ax = plt.subplots()
ax.plot(r, P)
ax.set_xlabel('Distance ($\AA$)')
plt.show()

# Save structure
ProEPR.save('MBP_L20R1_S238R1.pdb', SL1, SL2, protein=MBP)
```

### Mimicking MMM and MTSSLWizard
In addition to its own features ProEPR offers spin label modeling methods that mimic the popular MMM and MTSSLWizard 
modeling applications.

```python
import ProEPR
MBP = ProEPR.fetch('1omp')
SLmmm = ProEPR.SpinLabel.from_mmm('R1M', site=238, protein=MBP)
SLWiz = ProEPR.SpinLabel.from_wizard('R1M', site=238, protein=MBP,
                                     to_find=50, to_try=1000,      # Equivalent to 'quick' search, default is 'thorough'   
                                     vdw=3.4,  clashes=0,          # MTSSLWizard 'tight' setting, default is 'loose' 
                                     )
```

### Local repacking and off-rotamer sampling 
One of the benefits of ProERP is the variety and customizable nature of spin label modeling methods. This includes 
methods to repack a SpinLabel, and it's neighboring amino acids, and to sample side chain conformations that deviate from
canonical dihedral angles and fixed rotamer libraries.

```python
import ProEPR
MBP = ProEPR.fetch('1omp')

# Create a SpinLabel object using the MTSSLWizard 'Accessible Volume' Approach
SL1 = ProEPR.SpinLabel.from_wizard('R1C', site=20, chain='A', protein=MBP)

# Create a SpinLabel object by sampling off-rotamer dihedral conformations using the rotamer library as a prior 
SL2 = ProEPR.SpinLabel('R1C', site=238, chain='A', sample=2000, protein=MBP)

# Create a SpinLabel object from a ProEPR.repack trajectory
traj, de, SLs = ProEPR.repack(SL1, SL2, protein=MBP)
```
The repack function will perform a Markov chain Monte Carlo sampling repack of the spin labels, `SL1` and `SL2` and 
neighboring side chains, returning an `MDAnalysis.Universe` object containing all accepted structures of the MCMC 
trajectory, the energy function changes at each acceptance step and new SpinLabel objects attached to the lowest energy 
structure of the trajectory.

SpinLabel objects and neighboring side chains can be repacked using off-rotamer sampling by using the `off_rotamer=True`
option. In the event off rotamer sampling is being used for repacking, it is likely that the desired SpinLabel object is 
not the default rotamer library attached to the lowest energy structure, but instead the ensemble of side chains 
created in the MCMC sampling trajectory. This can be done using the `from_trajectory` class method. 

```python
# Create a SpinLabel object from a ProEPR.repack trajectory with off-rotamer sampling
traj, de, SLs = ProEPR.repack(SL1, SL2, protein=MBP, off_rotamer=True) 
SL1 = ProEPR.SpinLabel.from_trajectory(traj, site=238)
```

Off rotamer sampling can be controlled on a per dihedral basis when repacking with ProEPR by passing a list of bools to 
the off_rotamer variable. For example, passing `off_rotamer = [False, False, False, True, True]` will allow for off 
rotamer sampling of only $\chi_4$ and $\chi_5$.


### Mutating protein structures
Sometimes you don't want a whole rotamer library, you just want a protein structure mutated at a particular site with 
the most probable spin label conformer. This can be done easily with the `mutate` function.

```python
import ProEPR
MBP = ProEPR.fetch('1omp')
SL = ProEPR.SpinLabel('R1C', 238, protein=MBP)
MBP_S238R1 = ProEPR.mutate(MBP, SL)
ProEPR.save('MBP_S238R1.pdb', MBP_S238R1)
```

ProEPR can actually mutate several sites at once, and can mutate canonical amino acids as well.

```python
SL1 = ProEPR.SpinLabel('R1C', 20, protein=MBP)
SL2 = ProEPR.SpinLabel('R1C', 238, protein=MBP)
L284V = ProEPR.RotamerLibrary('VAL', 284, protein=MBP)
```

 Mtating adjacent sites is best done with the `repack` function to avoid clashes between SpinLabels/RotamerLibraries. 
This will return a trajectory which can be used to pick the last or lowest energy frame as your mutated protein.

```python
MBP_L284V_L20R1_S238R1, _, _ = ProEPR.repack(SL1, SL2, L284V, protein=MBP)
```

### Adding user defined spin labels
Site directed spin labels, and other site directed labels, are constantly being developed. To this end ProEPR makes it 
easy to add user spin labels. To add a user defined spin label, all that is needed is (1) a pdb file of the spin label
(2) A list of the rotatable dihedral bonds, and (3) a list of the atoms where the spin is.

```python
ProEPR.add_label(name='TRT',
                 pdb='test_data/trt.pdb',
                 dihedral_atoms=[['CB', 'SG', 'SD', 'CAD'],
                                 ['SG', 'SD', 'CAD', 'CAE'],
                                 ['SD', 'CAD', 'CAE', 'OAC']],
                 spin_atoms='CAQ')
```

User defined labels can be constructed from a single state pdb file or a multi-state pdb file. If constructed from a 
single state pdb file a list of dihedral angles and weights can be passed via the `dihedrals` and `weigts` keyword
arguments. For each set of dihedral angles, ProEPR create a rotamer and store the whole library using the specified 
name. Once a label is added it can be used the same as any other label. e.g.

```python
ProEPR.SpinLable('TRT', site=238, protein=MBP, sample=5000)
```

For more information on how to use ProEPR as a python module, see [examples](#examples/)
