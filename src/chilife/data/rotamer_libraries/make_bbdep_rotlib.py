from functools import partial
import numpy as np
import pandas as pd
import pickle
from glob import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D


"""This utility will construct all rotamer library in the proper coordinate frame when provided with 1) and deer2bb 
rotamer library in pickle format (Same as pymol) and 2) pdb files with the base structure of all rotamers of interest
PDB files and deer2bb_bb_ind.pkl names must be the same with the exception of the pdb extension. This application also
will plot the NO midpoint coordinates and N, CA, CB coordinates so that the user can review the coordinate 
transformations to ensure nothing went wrong"""


def main():
    if os.path.exists('file.pdb'):
        os.remove('file.pdb')
    names = ['resn', 'Phi', 'Psi', 'count', 'r1', 'r2', 'r3', 'r4', 'r5', 'Prob',
             'chi1Val', 'chi2Val', 'chi3Val', 'chi4Val', 'chi5Val', 'chi1Sig',
             'chi2Sig', 'chi3Sig', 'chi4Sig', 'chi5Sig']
    bbdep_lib = pd.read_csv('R1C.rotlib', delim_whitespace=True, names=names)
    bbdep_lib['resn'] = 'R1C'
    bbdep_lib_dict = {}
    rotamer_name = 'R1C'
    pdb_filename = rotamer_name + '.pdb'
    col_names = ['ATOM', 'atom number', 'atom name', 'residue name', 'chain', 'residue number',
                 'x', 'y', 'z', 'occupancy', 'Bfactor', 'atom type']
    coords = ['x', 'y', 'z']
    print(pdb_filename)
    struct = pd.read_csv(pdb_filename,
                         dtype={'ATOM': str, 'atom number': int, 'atom name': str, 'residue name': str,
                                'chain': str, 'residue number': int, 'x': float, 'y': float, 'z': float,
                                'occupancy': float, 'Bfactor': float, 'atom type': str},
                         delim_whitespace=True, names=col_names)

    atom_names = struct['atom name'].values
    atom_type = struct['atom type'].values
    dihedral_definitions = [('N', 'CA', 'CB', 'SG'),
                            ('CA', 'CB', 'SG', 'SD'),
                            ('CB', 'SG', 'SD', 'CE'),
                            ('SG', 'SD', 'CE', 'C3'),
                            ('SD', 'CE', 'C3', 'C2')]

    # Align rotamer backbone to defined coordinate system
    N = struct[struct['atom name'] == 'N'][coords].values[0]
    C = struct[struct['atom name'] == 'C'][coords].values[0]
    CA = struct[struct['atom name'] == 'CA'][coords].values[0]

    ori, mx = rosetta_local_mx(N, CA, C)

    struct[coords] -= ori
    struct[coords] = struct[coords].values.dot(mx)

    groups = bbdep_lib.groupby(['resn', 'Phi', 'Psi'])
    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        libs = tqdm(pool.imap(partial(all_structs,
                          struct=struct,
                          dihedral_definitions=dihedral_definitions,
                          coords=coords), groups), total=1369)

        lib = {results[0]: results[1:] for results in libs}

    np.savez(rotamer_name + '_bbdep_rotlib.npz', lib=lib, atom_types=atom_type,
              atom_names=atom_names, dihedral_atoms=dihedral_definitions)

def all_structs(grp, struct, dihedral_definitions, coords):
    key, grp = grp
    weights = grp.Prob.values
    rot_lib = grp[['chi1Val', 'chi2Val', 'chi3Val', 'chi4Val', 'chi5Val']].values
    rotamer_coords = [get_struct(rotamer, struct, dihedral_definitions, coords) for rotamer in rot_lib]
    rotamer_coords = np.array(rotamer_coords)
    return key, rot_lib, rotamer_coords, weights

def get_struct(rotamer, struct0, dihedral_definitions, coords):
    struct = struct0.copy()
    for angle, dihedral in zip(rotamer, dihedral_definitions):
        mobile_idx = int(struct.index.where(struct['atom name'] == dihedral[-2]).dropna().values[0])
        mobile = struct.iloc[mobile_idx:-2, 6:9].values
        p = struct[struct['atom name'].isin(dihedral)][coords].values
        struct.iloc[mobile_idx:-2, 6:9] = set_dihedral(p, angle, mobile)

    return struct[coords].values

def rosetta_local_mx(N, CA, C):
    """
    Calculates a translation and rotation matrix to transform an amino acid from the global coordinate frame to the
    local coordinate frame as defined by the rosetta convention. The analogous transformation can be found in:
    src/numeric/HomogeneousTransform.cc

    :param N: numpy ndarray (1x3)
        Backbone nitrogen coordinates

    :param CA: numpy ndarray (1x3)
        Backbone Calpha carbon coordinates

    :param C: numpy ndarray (1x3)
        Backbone carbonyl carbon coordinates

    :return (origin, rotation_matrix) : (numpy ndarray (1x3), numpy ndarra (3x3))
        origin: new origin position in 3 dimensional space
        rotation_matrix: rotation  matrix to rotate spin label to
    """

    # Tranform coordinates such that the CA atom is at the origin
    p1 = N - CA
    p2 = C - CA
    p3 = CA - CA

    # Define Z-axis unit vector
    zaxis = p3 - p2
    zaxis = zaxis / np.linalg.norm(zaxis)

    # Calculate C-->N vector
    p21 = p1 - p2

    # Calculate magnitude of C-->N vector in Z direction
    p21_z_comp = p21.dot(zaxis)

    # Subtract Z component from C-->N vector to get a vector orthogonal to Z direction in the N-Ca-C plane
    yaxis = p21 - p21_z_comp * zaxis
    yaxis = yaxis / np.linalg.norm(yaxis)

    # Calculate x axis as cross product of y and z axis
    xaxis = np.cross(yaxis, zaxis)

    # Create rotation matrix to rotate any atoms to new coordinate frame
    rotation_matrix = np.linalg.inv(np.array([xaxis, yaxis, zaxis]))

    # Set origin at calpha
    origin = CA

    return origin, rotation_matrix


def get_dihedral(p):
    """
    Calculates dihedral of a given set of atoms, p = [0, 2, 3, 3] using Praxeolitic's formula (stackoverflow). Returns
    value in degrees.

                    3
         ------>  /
        1-------2
      /
    0
    """

    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    # Define vectors from coordinates
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize dihedral bond vector
    b1 /= np.linalg.norm(b1)

    # Calculate dihedral projections orthogonal to the bond vector
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # Calculate angle between projections
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    # Return as degrees
    return np.degrees(np.arctan2(y, x))


def set_dihedral(p, angle, mobile):
    """
    Sets the dihedral angle by rotating all 'mobile' atoms from their current position about the dihedral bond defined
    by the four atoms in p. Dihedral will be set to the value of 'angle' in degrees

    :param p: array-like int
        Atom coordinates that define dihedral to rotate about

    :param angle: float
        New angle to set the dihedral to (degrees).

    :param mobile
        atoms to move by setting dihedral

    """
    current = get_dihedral(p)
    angle = angle - current
    angle = np.deg2rad(angle)

    ori = p[1]
    mobile -= ori
    v = p[2] - p[1]
    v /= np.linalg.norm(v)
    R = get_dihedral_rotation_matrix(angle, v)

    new_mobile = R.dot(mobile.T).T + ori

    return new_mobile


def get_dihedral_rotation_matrix(theta, v):
    """
    Build a matrix that will rotate any coordinates about a vector, v, by theta in radians

    :param theta: float
        Rotation  angle in radians

    :param v: numpy ndarray (1x3)
        Three dimensional vector to rotate about

    :return rotation_matrix:
        Matrix that will rotate
    """

    # Normalize input vector
    v = v / np.linalg.norm(v)

    # Compute Vx matrix
    Vx = np.zeros((3, 3))
    Vx[[2, 0, 1], [1, 2, 0]] = v
    Vx -= Vx.T

    # Rotation matrix. See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rotation_matrix = np.identity(3) * np.cos(theta) + np.sin(theta) * Vx + (1 - np.cos(theta)) * np.outer(v, v)

    return rotation_matrix

pdb_formatter = {'ATOM': '{:<5s}'.format, 'atom number': '{:>5d}'.format, 'atom name': '{:<4s} '.format,
                 'residue name': '{:<3s}'.format, 'chain': '{:<1s}'.format, 'residue number': '{:>4d} '.format,
                 'x': '{:>8.3f}'.format, 'y': '{:>8.3f}'.format, 'z': '{:>8.3f}'.format, 'occupancy': '{:>6.2f}'.format,
                 'Bfactor': '{:>6.2f}'.format, 'atom type': '{:>2s}'.format}

if __name__ == '__main__':
    main()
