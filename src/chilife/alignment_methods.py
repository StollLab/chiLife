import numpy as np
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
    """Calculates a rotation matrix and translation to transform an amino acid from the local coordinate frame to the
    global coordinate frame for a given residue with backbone coordinates (N, C, CA). Principal axis bisects the
    N->Ca->C angle.

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
    # Define new Z axis that bisects N<--CA-->C angle
    CA_N = N - CA
    CA_N /= np.linalg.norm(CA_N)
    CA_C = C - CA
    CA_C /= np.linalg.norm(CA_C)
    zaxis = CA_N + CA_C
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

    if method.__name__ == "fit_alignment":
        N1, CA1, C1 = rotamer_ensemble.backbone
        N2, CA2, C2 = rotamer_ensemble.protein.select_atoms(
            f"segid {rotamer_ensemble.chain} and "
            f"resnum {rotamer_ensemble.site} "
            f"and name N CA C and not altloc B"
        ).positions
        return np.array([[N1, N2], [CA1, CA2], [C1, C2]])

    elif kind == "local":
        return rotamer_ensemble.backbone

    elif kind == "global":
        return rotamer_ensemble.protein.select_atoms(
            f"segid {rotamer_ensemble.chain} and "
            f"resnum {rotamer_ensemble.site} "
            f"and name N CA C and not altloc B"
        ).positions
