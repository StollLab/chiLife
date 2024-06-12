import numpy as np
import igraph as ig

def normalize_angles(angles):
    """

    Parameters
    ----------
    angles

    Returns
    -------

    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def angle_dist(angle1, angle2):
    """

    Parameters
    ----------
    angle1
    angle2

    Returns
    -------

    """
    diff = angle1 - angle2
    return np.arctan2(np.sin(diff), np.cos(diff))


def simple_cycle_vertices(graph: ig.Graph):
    found_cycle_edges = graph.fundamental_cycles()
    found_cycle_nodes = []

    for cycle in found_cycle_edges:
        cyverts = set()
        for edge in graph.es(cycle):
            cyverts.update(edge.tuple)

        found_cycle_nodes.append(sorted(cyverts))
    return found_cycle_nodes