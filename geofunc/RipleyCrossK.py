import numpy as np

def RipleyCrossK(x_min, x_max, y_min, y_max, ligand_distr, receptor_distr, radii=None):
    """
    Calculate Ripley's Cross K function for two spatial point distributions (ligand and receptor).

    Parameters:
    x_min, x_max, y_min, y_max (float): The boundaries of the study area.
    ligand_distr (array-like): Coordinates of points for the ligand distribution.
    receptor_distr (array-like): Coordinates of points for the receptor distribution.
    radii (array-like, optional): Radii at which to calculate the function. If None, defaults to 200 equally spaced points.

    Returns:
    array: Values of Ripley's Cross K function at specified radii.
    """

    area = (x_max - x_min) * (y_max - y_min)
    if radii is None:
        radii = np.linspace(0, x_max - x_min, 200).reshape(200, 1)

    npts = len(ligand_distr)  # Number of points in the ligand distribution
    cross_npts = len(receptor_distr)  # Number of points in the receptor distribution

    # Calculate pairwise distances
    diff = np.array([np.abs(l - r) for l in ligand_distr for r in receptor_distr])

    # Calculate weights for edge correction
    hor_dist = [min(x_max - r[0], r[0] - x_min) for r in receptor_distr] * npts
    ver_dist = [min(y_max - r[1], r[1] - y_min) for r in receptor_distr] * npts
    dist = np.hypot(diff[:, 0], diff[:, 1])
    dist_ind = dist <= np.hypot(hor_dist, ver_dist)
    w1 = (1 - (np.arccos(np.minimum(ver_dist, dist) / dist) + np.arccos(np.minimum(hor_dist, dist) / dist)) / np.pi)
    w2 = (3 / 4 - 0.5 * (np.arccos(ver_dist / dist * ~dist_ind) + np.arccos(hor_dist / dist * ~dist_ind)) / np.pi)
    weight = dist_ind * w1 + ~dist_ind * w2

    # Calculate Ripley's Cross K
    ripley = [np.nansum((dist < r) / weight) for r in radii.flatten()]
    ripley = area * np.array(ripley) / (npts * cross_npts)

    return ripley
