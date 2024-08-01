import numpy as np
import igraph as ig

def normalize_angles(angles):
    """
    reduce angles (in radians) to +/- pi.

    Parameters
    ----------
    angles : ArrayLike
        Angles to noramlize/reduce

    Returns
    -------
    angles : ArrayLike
        Reduced/normalized angles
    """

    return np.arctan2(np.sin(angles), np.cos(angles))


def angle_dist(angle1, angle2):
    """
    Get the minimum difference (distance) between two angles

    Parameters
    ----------
    angle1 : ArrayLike
        first set of angles (in radians)

    angle2: ArrayLike
        second set of angles (in radians)

    Returns
    -------
    distance : ArrayLike
        The angular distance (in radians) between angle1 and angle2 on the unit circle
    """
    diff = angle1 - angle2
    return np.arctan2(np.sin(diff), np.cos(diff))


def simple_cycle_vertices(graph: ig.Graph):
    """
    Find the vertices of all simple cycles in the graph

    Parameters
    ----------
    graph: ig.Graph
        The graph to find the cycle vertices of.

    Returns
    -------
    found_cycle_nodes : List[List[int]]
        A list of all simple cycles in the graph. Each simple cycle is composed of a list of vertices that belong to
        that cycle.
    """

    found_cycle_edges = graph.fundamental_cycles()
    found_cycle_nodes = []

    for cycle in found_cycle_edges:
        cyverts = set()
        for edge in graph.es(cycle):
            cyverts.update(edge.tuple)

        found_cycle_nodes.append(sorted(cyverts))
    return found_cycle_nodes