"""
TODO: test log_cdf against the (slower) cdf_quad
"""

import astropy.units as u
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .integrate import log_simpson


def get_log_cdf(log_density, grid, args=()):
    grid_dz = grid[1] - grid[0]

    normalize, err = quad(
        lambda z: np.exp(log_density(z, *args)),
        grid[0], grid[-1], epsabs=1e-10
    )
    log_normalize = np.log(normalize)

    log_pdf = log_density(grid, *args) - log_normalize
    log_cdf = np.array([
        log_simpson(log_pdf[:i], dx=grid_dz, even='last')
        for i in range(1, len(grid) + 1)
    ])

    return grid, log_cdf


def get_cdf_quad(log_density, grid_lim=[-5, 5]*u.kpc, grid_size=1024, args=()):
    zgrid = np.linspace(*grid_lim.to_value(u.pc), grid_size)
    cdf = [quad(lambda z: np.exp(log_density(z, *args)), zgrid[0], zgrid[i])[0]
           for i in range(1, len(zgrid))]
    return zgrid * u.pc, np.concatenate(([0.], cdf))


def invt_sample_z(log_density, size=1, rng=None, **log_cdf_kwargs):
    """
    Inverse transform sampling to generate "true" z values for simulation:
    """
    if rng is None:
        rng = np.random.default_rng()

    zgrid, log_cdf = get_log_cdf(log_density, **log_cdf_kwargs)
    cdf = np.exp(log_cdf)

    interp_f = interp1d(cdf, zgrid, kind='cubic')
    z_samples = interp_f(rng.uniform(cdf.min(), cdf.max(), size=size))
    return z_samples
