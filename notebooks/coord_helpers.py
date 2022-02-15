from astropy.coordinates.matrix_utilities import (
    rotation_matrix,
    matrix_product
)


def get_M(sgrA_star, zsun, roll):
    # This matrix goes from "Galactic" to Shmagalactic
    lat_mat = rotation_matrix(-sgrA_star.lat, 'y')
    lon_mat = rotation_matrix(sgrA_star.lon, 'z')
    roll_mat = rotation_matrix(roll, 'x')
    
    # construct transformation matrix and use it
    R = matrix_product(roll_mat, lat_mat, lon_mat)

    # Now need to account for tilt due to Sun's height above the plane
    z_d = zsun / sgrA_star.distance
    H = rotation_matrix(-np.arcsin(z_d), 'y')

    # compute total matrices
    M = matrix_product(H, R)

    return M


# schmagal = Galactocentric but at the solar position
# gal = Galactic
def gal_to_schmagal(xyz, sgrA_star, zsun, roll):
    """
    Transform from Galactic to Schmagalactic cartesian coordinates.
    
    Galactic = the usual
    Schmagalactic = aligned with local Galactic disk, z at midplane, x to Galactic center
    """
    M = get_M(sgrA_star, zsun, roll)
    new_xyz = M @ xyz - ([0, 0, zsun.value] * zsun.unit)[:, None]
    return new_xyz
    

def schmagal_to_gal(xyz, sgrA_star, zsun, roll):
    """
    Transform from Schmagalactic to Galactic cartesian coordinates.
    
    Galactic = the usual
    Schmagalactic = aligned with local Galactic disk, z at midplane, x to Galactic center
    """
    MT = get_M(sgrA_star, zsun, roll).T
    new_xyz = MT @ (xyz + ([0, 0, zsun.value] * zsun.unit)[:, None])
    return new_xyz
