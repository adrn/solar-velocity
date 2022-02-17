import astropy.units as u
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


def invt_sample_z(density, size=1, 
                  grid_lim=[-5, 5]*u.kpc, grid_size=256, 
                  args=(), rng=None):
    """
    Inverse transform sampling to generate "true" z values for simulation:
    """
    if rng is None:
        rng = np.random.default_rng()
    
    zgrid = np.linspace(*grid_lim.to_value(u.pc), grid_size)
    pdf = density(zgrid, *args)
    cdf = np.array([
        quad(density, -np.inf, zz, args=args, limit=1024, epsabs=1e-10)[0] 
        for zz in zgrid
    ])
    
    interp_f = interp1d(cdf, zgrid, kind='cubic')
    z_samples = interp_f(rng.uniform(cdf.min(), cdf.max(), size=size))
    return z_samples
