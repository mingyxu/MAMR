"""
Contains methods to manipulate hex bins
"""

def hex_neighbors(hex_bin, hex_attr):
    """
    Returns a set of neighboring hex bins
    :param hex_bin:
    :param hex_attr:
    :return neighbors:
    """
    neighbor_dirs = ['north_east_neighbor',
                     'north_neighbor',
                     'north_west_neighbor',
                     'south_east_neighbor',
                     'south_neighbor',
                     'south_west_neighbor']
    candidates = hex_attr[hex_attr['hex_id']==hex_bin][neighbor_dirs].values[0]
    neighbors = []
    for hex_bin in candidates:
        try:
            neighbors.append(int(hex_bin))
        except:
            pass
    return set(neighbors)


def hex_neighborhood(hex_bin, hex_attr, radius):
    """
    Returns a set of hex bins in given radius
    :param hex_bin:
    :param hex_attr:
    :param radius:
    :return neighborhood:
    """
    candidate_bins = set([hex_bin])
    neighborhood = set([hex_bin])
    for r in range(radius):
        new_candidate_bins = set([])
        for hex_bin in candidate_bins:
            neighbors = hex_neighbors(hex_bin, hex_attr)
            new_candidate_bins = new_candidate_bins.union(neighbors)
            neighborhood = neighborhood.union(neighbors)
        candidate_bins = new_candidate_bins
    return list(neighborhood)
