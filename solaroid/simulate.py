import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from .integrate import log_simpson


def invt_sample_z(log_density, size=1,
                  grid_lim=[-5, 5]*u.kpc, grid_size=1024,
                  args=(), rng=None):
    """
    Inverse transform sampling to generate "true" z values for simulation:
    """
    if rng is None:
        rng = np.random.default_rng()

    zgrid = np.linspace(*grid_lim.to_value(u.pc), grid_size)
    grid_step = zgrid[1] - zgrid[0]
    zgrid = np.concatenate((
        np.arange(zgrid[0] - 3 * grid_step, zgrid[0], grid_step),
        zgrid
    ))

    log_pdf = log_density(zgrid, *args)

    log_cdf = np.array([
        log_simpson(log_pdf[:i], x=zgrid[:i])
        for i in range(3, len(zgrid))
    ])
    cdf = np.exp(log_cdf)

    interp_f = interp1d(cdf, zgrid[3:], kind='cubic')
    z_samples = interp_f(rng.uniform(cdf.min(), cdf.max(), size=size))
    return z_samples
