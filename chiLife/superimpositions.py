import numpy as np

superimpositions = {}


def superimposition(func):
    """Decorator to add function to the superimpositions dictionary"""
    superimpositions[func.__name__.split("_")[0]] = func
    return func


@superimposition
def rosetta_superimposition(N, CA, C):
    """
        Calculates a rotation matrix and translation to transform an amino acid from the local coordinate frame to the
    global coordinate frame for a given residue with backbone coordinates (N, C, CA). Principal acis is the C->Ca Bond.

    :param N: numpy ndarray (1x3)
        Backbone nitrogen coordinates

    :param CA: numpy ndarray (1x3)
        Backbone Calpha carbon coordinates

    :param C: numpy ndarray (1x3)
        Backbone carbonyl carbon coordinates

    :return (rotation_matrix, origin) : (numpy ndarray (1x3), numpy ndarray (3x3))
        rotation_matrix: rotation  matrix to rotate spin label to
        origin: new origin position in 3 dimensional space
    """
    # Define new Z axis from  C-->CA
    zaxis = CA - C
    zaxis = zaxis / np.linalg.norm(zaxis)

    # Define new y axis
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


@superimposition
def bisect_superimposition(N, CA, C):
    """
        Calculates a rotation matrix and translation to transform an amino acid from the local coordinate frame to the
    global coordinate frame for a given residue with backbone coordinates (N, C, CA). Principal axis bisects the
    N->Ca->C angle.

    :param N: numpy ndarray (1x3)
        Backbone nitrogen coordinates

    :param CA: numpy ndarray (1x3)
        Backbone Calpha carbon coordinates

    :param C: numpy ndarray (1x3)
        Backbone carbonyl carbon coordinates

    :return (rotation_matrix, origin) : (numpy ndarray (1x3), numpy ndarray (3x3))
        rotation_matrix: rotation  matrix to rotate spin label to
        origin: new origin position in 3 dimensional space
    """
    # Define new Z axis that bisects N<--CA-->C angle
    CA_N = N - CA
    CA_N /= np.linalg.norm(CA_N)
    CA_C = C - CA
    CA_C /= np.linalg.norm(CA_C)
    zaxis = CA_N + CA_C
    zaxis = zaxis / np.linalg.norm(zaxis)

    # Define new y axis
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


@superimposition
def mmm_superimposition(N, CA, C):
    """
    Calculates a rotation matrix and translation to transform an amino acid from the local coordinate frame to the
    global coordinate frame for a given residue with backbone coordinates (N, C, CA). Principal axis is defined
    along the CA->N bond.

    :param N: numpy ndarray (1x3)
        Backbone nitrogen coordinates

    :param CA: numpy ndarray (1x3)
        Backbone Calpha carbon coordinates

    :param C: numpy ndarray (1x3)
        Backbone carbonyl carbon coordinates

    :return (rotation_matrix, origin) : (numpy ndarray (1x3), numpy ndarray (3x3))
        rotation_matrix: rotation  matrix to rotate spin label to
        origin: new origin position in 3 dimensional space
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


@superimposition
def fit_superimposition(N, CA, C):
    inp = np.array([N, CA, C])

    # Check if input is transposed
    for ar in inp:
        if ar.shape[1] == 2 and ar.shape[0] == 3:
            ar = ar.T

    # Check if input is the appropriate shape
    if len(N) != len(CA) != len(C) != 2:
        raise ValueError(
            "fit_superimposition takes 3 sets of 2 coordinates. The first coordinate should be from the "
            "mobile residue and the second coordinate should be from the target residue."
        )

    raise NotImplementedError(
        "Superimposition by minimizing RMSD of the backbone coordinates it not yet implemented"
    )


def parse_backbone(rotamer_library, kind):
    """
    Extract appropriate backbone information to make a rotation matrix using the method provided

    :param rotamer_library: RotamerLibrary
        RotamerLibrary that the rotation matrix will operate on. If using the `fit` method, the rotamer library must
        have a `protein` feature.

    :return N, CA, C: tuple
        Numpy arrays of N, CA and C coordinates of the rotamer library backbone. If using method `fit` arrays are 2x3
        with the first coordinate as the rotamer library backbone and the second as the protein site backbone.
    """
    method = rotamer_library.superimposition_method

    if method in {"fit"}:
        N1, CA1, C1 = rotamer_library.backbone
        N2, CA2, C2 = rotamer_library.protein.select_atoms(
            f"segid {rotamer_library.chain} and "
            f"resnum {rotamer_library.site} "
            f"and name N CA C and not altloc B"
        ).positions
        return np.array([[N1, N2], [CA1, CA2], [C1, C2]])
    elif kind == "local":
        return rotamer_library.backbone
    elif kind == "global":
        return rotamer_library.protein.select_atoms(
            f"segid {rotamer_library.chain} and "
            f"resnum {rotamer_library.site} "
            f"and name N CA C and not altloc B"
        ).positions
