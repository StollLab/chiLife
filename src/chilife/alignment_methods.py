from typing import Union, List, Tuple

import numpy as np
from numpy.typing import ArrayLike
alignment_methods = {}


def alignment_method(func):
    """Decorator to add function to the superimpositions dictionary

    Parameters
    ----------
    func : callable
        Function that performs a alignment_method.

    Returns
    -------
        Unmodified original function.
    """
    alignment_methods[func.__name__.split("_")[0]] = func
    return func


@alignment_method
def rosetta_alignment(N, CA, C):
    """Calculates a rotation matrix and translation to transform an amino acid from the local coordinate frame to the
    global coordinate frame for a given residue with backbone coordinates (N, C, CA). Principal axis is the C->Ca Bond.

    Parameters
    ----------
    N : numpy.ndarray (1x3)
        Backbone nitrogen coordinates.
    CA : numpy.ndarray (1x3)
        Backbone alpha carbon coordinates.
    C : numpy.ndarray (1x3)
        Backbone carbonyl carbon coordinates.

    Returns
    -------
    rotation_matrix : numpy ndarray (3x3)
        Rotation  matrix to rotate spin label to achieve the correct orientation.
    origin : numpy.ndarray (1x3)
        New origin position in 3-dimensional space.
    """
    # Define new Z axis from  C-->CA
    zaxis = CA - C
    zaxis = zaxis / np.linalg.norm(zaxis)

    # Define new y-axis
    yaxis_plane = N - C
    z_comp = yaxis_plane.dot(zaxis)
    yaxis = yaxis_plane - z_comp * zaxis
    yaxis = yaxis / np.linalg.norm(yaxis)

    # Define new x axis
    xaxis = np.cross(yaxis, zaxis)

    # Create rotation matrix
    rotation_matrix = np.array([xaxis, yaxis, zaxis])
    origin = CA
    return rotation_matrix, origin


@alignment_method
def bisect_alignment(N, CA, C):
    """Calculates the rotation matrix and translation to transform an amino acid from the local coordinate frame to the
    global coordinate frame for a given residue with backbone coordinates (N, C, CA). The principal z axis bisects the
    N-CA-C angle, and y is in the N-CA-C plane perpendicular to z, approximately pointing from C to N.

    Parameters
    ----------
    N : numpy.ndarray (1x3)
        Backbone nitrogen coordinates.
    CA : numpy.ndarray (1x3)
        Backbone alpha carbon coordinates.
    C : numpy.ndarray (1x3)
        Backbone carbonyl carbon coordinates.

    Returns
    -------
    rotation_matrix : numpy .darray (3x3)
        Rotation matrix to rotate spin label to achieve the correct orientation.
    origin : numpy.ndarray (1x3)
        New origin position in 3-dimensional space.
    """
    # Define new z axis: bisector of N-CA-C angle
    CA_N = N - CA
    CA_N /= np.linalg.norm(CA_N)
    CA_C = C - CA
    CA_C /= np.linalg.norm(CA_C)
    zaxis = CA_N + CA_C
    zaxis = zaxis / np.linalg.norm(zaxis)

    # Define new y axis: perpendicular to z in N-CA-C plane
    yaxis = CA_N - CA_C
    yaxis = yaxis / np.linalg.norm(yaxis)

    # Define new x axis: perpendicular to y and z
    xaxis = np.cross(yaxis, zaxis)

    # Create rotation matrix
    rotation_matrix = np.array([xaxis, yaxis, zaxis])
    
    origin = CA
    
    return rotation_matrix, origin


@alignment_method
def mmm_alignment(N, CA, C):
    """Calculates a rotation matrix and translation to transform an amino acid from the local coordinate frame to the
    global coordinate frame for a given residue with backbone coordinates (N, C, CA). Principal axis is defined
    along the CA->N bond.

    Parameters
    ----------
    N : numpy.ndarray (1x3)
        Backbone nitrogen coordinates.
    CA : numpy.ndarray (1x3)
        Backbone alpha carbon coordinates.
    C : numpy.ndarray (1x3)
        Backbone carbonyl carbon coordinates.

    Returns
    -------
    rotation_matrix : numpy ndarray (3x3)
        Rotation  matrix to rotate spin label to achieve the correct orientation.
    origin : numpy.ndarray (1x3)
        New origin position in 3-dimensional space.
    """
    xaxis = N - CA
    xaxis /= np.linalg.norm(xaxis)

    yp = C - CA
    yp /= np.linalg.norm(yp)
    zaxis = np.cross(xaxis, yp)
    zaxis /= np.linalg.norm(zaxis)

    yaxis = np.cross(zaxis, xaxis)

    rotation_matrix = np.array([xaxis, yaxis, zaxis])

    origin = CA

    return rotation_matrix, origin


@alignment_method
def fit_alignment(N, CA, C):
    """Superimpose the residues such that the root mean squared deviation of the backbone atoms is minimized.

    Parameters
    ----------
    N : numpy.ndarray (2x3)
        Backbone nitrogen coordinates.
    CA : numpy.ndarray (2x3)
        Backbone alpha carbon coordinates.
    C : numpy.ndarray (2x3)
        Backbone carbonyl carbon coordinates.

    Returns
    -------
    rotation_matrix : numpy ndarray (3x3)
        Rotation  matrix to rotate spin label to achieve the correct orientation.
    origin : numpy.ndarray (1x3)
        New origin position in 3-dimensional space.
    """

    # Check if input is transposed
    lst = []
    for ar in (N, CA, C):
        if ar.shape[1] == 2 and ar.shape[0] == 3:
            ar = ar.T
        lst.append(ar)

    N, CA, C = lst

    inp0 = np.array([N[0], CA[0], C[0]])
    target0 = np.array([N[1], CA[1], C[1]])

    # Check if input is the appropriate shape
    if len(N) != len(CA) != len(C) != 2:
        raise ValueError(
            "fit_alignment takes 3 sets of 2 coordinates. The first coordinate should be from the "
            "mobile residue and the second coordinate should be from the target residue."
        )

    origin = np.mean(inp0, axis=0)

    inp = inp0 - origin
    target = target0 - np.mean(target0, axis=0)

    inp_normal = np.cross(inp[0]-inp[1], inp[2] - inp[1])
    inp_normal /= np.linalg.norm(inp_normal)
    inp = np.vstack([inp, inp_normal])

    target_normal = np.cross(target[0]-target[1], target[2]-target[1])
    target_normal /= np.linalg.norm(target_normal)
    target = np.vstack([target, target_normal])

    H = inp.T @ target
    U, S, V = np.linalg.svd(H)
    d = np.sign(np.linalg.det(V @ U.T))
    new_S = np.eye(3)
    new_S[2, 2] = d
    rotation_matrix = V @ U.T

    raise NotImplementedError(
        "Superimposition by minimizing RMSD of the backbone coordinates it not yet implemented"
    )

    return rotation_matrix, origin


def parse_backbone(rotamer_ensemble, kind):
    """Extract appropriate backbone information to make a rotation matrix using the method provided

    Parameters
    ----------
    rotamer_ensemble : RotamerEnsemble
        The RotamerEnsemble object that the rotation matrix will operate on. If using the `fit` method, the rotamer
        ensemble must have a `protein` feature.
    kind : str
        Specifies if the backbone is for the rotamer ensemble (local) or the protein (global)

    Returns
    -------
    N, CA, C: tuple
        Numpy arrays of N, CA and C coordinates of the rotamer ensemble backbone. If using method `fit` arrays are 2x3
        with the first coordinate as the rotamer ensemble backbone and the second as the protein site backbone.
    """
    method = rotamer_ensemble.alignment_method
    aln_atoms = " ".join(rotamer_ensemble.aln_atoms)
    if method.__name__ == "fit_alignment":
        N1, CA1, C1 = rotamer_ensemble.aln
        N2, CA2, C2 = rotamer_ensemble.protein.select_atoms(
            f"segid {rotamer_ensemble.chain} and "
            f"resid {rotamer_ensemble.site} "
            f"and name {aln_atoms} and not altloc B"
        ).positions
        return np.array([[N1, N2], [CA1, CA2], [C1, C2]])

    elif kind == "local":
        return rotamer_ensemble.aln

    elif kind == "global":
        return rotamer_ensemble.protein.select_atoms(
            f"segid {rotamer_ensemble.chain} and "
            f"resid {rotamer_ensemble.site} "
            f"and name {aln_atoms} and not altloc B"
        ).positions


def local_mx(*p, method: Union[str, callable] = "bisect") -> Tuple[ArrayLike, ArrayLike]:
    """Calculates a translation vector and rotation matrix to transform a set of coordinates from the global
    coordinate frame to a local coordinate frame defined by ``p`` , using the specified method.

    Parameters
    ----------
    p : ArrayLike
        3D coordinates of the three points defining the coordinate system (Usually N, CA, C).
    method : str, callable
        Method to use for generation of rotation matrix

    Returns
    -------
    origin : np.ndarray
        Cartesian coordinate of the origin to be subtracted from the coordinates before applying the rotation matrix.
    rotation_matrix : np.ndarray
        Rotation matrix to transform a set of coordinates to the local frame defined by p and the selected method.
    """

    if isinstance(method, str):
        method = alignment_methods[method]

    p1, p2, p3 = p

    if method.__name__ == 'fit_alignment':
        rotation_matrix, _ = method(p1, p2, p3)
        origin = np.mean([p1[0], p2[0], p3[0]], axis=0)
    else:
        # Transform coordinates such that the CA atom is at the origin
        p1n = p1 - p2
        p3n = p3 - p2
        p2n = p2 - p2

        origin = p2

        # Local Rotation matrix is the inverse of the global rotation matrix
        rotation_matrix, _ = method(p1n, p2n, p3n)

    rotation_matrix = rotation_matrix.T

    return origin, rotation_matrix


def global_mx(*p: ArrayLike, method: Union[str, callable] = "bisect") -> Tuple[ArrayLike, ArrayLike]:
    """Calculates a translation vector and rotation matrix to transform the a set of coordinates from the local
    coordinate frame to the global coordinate frame using the specified method.

    Parameters
    ----------
    p : ArrayLike
        3D coordinates of the three points used to define the new coordinate system (Usually N, CA, C)
    method : str
        Method to use for generation of rotation matrix

    Returns
    -------
    rotation_matrix : np.ndarray
        Rotation matrix to be applied to the set of coordinates before translating
    origin : np.ndarray
        Vector to be added to the coordinates after rotation to translate the coordinates to the global frame.
    """

    if isinstance(method, str):
        method = alignment_methods[method]

    if method.__name__ == 'fit_alignment':
        p = [pi[::-1] for pi in p]

    rotation_matrix, origin = method(*p)

    return rotation_matrix, origin
