import numpy as np
from coord_helpers import gal_to_schmagal, schmagal_to_gal


def ln_normal(x, mu, std):
    return -0.5 * (x-mu)**2 / std**2 - 0.5*np.log(2*np.pi) - np.log(std)


def ln_uniform(x, a, b):
    return -np.log(b - a)


def ln_two_sech2(z, h1, h2, f):
    lnterm1 = np.log(f) - 2 * np.log(np.cosh(z / (2 * h1))) - np.log(4 * h1)
    lnterm2 = np.log(1 - f) - 2 * np.log(np.cosh(z / (2 * h2))) - np.log(4 * h2)
    return np.logaddexp(lnterm1, lnterm2)


def ln_prob_density(xyz, ln_N0, z_args, ln_x_prob, x_args, ln_y_prob, y_args):
    return (ln_N0 + 
            ln_x_prob(xyz[0], *x_args) + 
            ln_y_prob(xyz[1], *y_args) + 
            ln_z_prob(xyz[2], *z_args))


def log_integrand(l, b, d, x_args, y_args, z_args, gal_args):
    l, b, d = map(np.array, [l, b, d])
    
    x = d * np.cos(l) * np.cos(b)
    y = d * np.sin(l) * np.cos(b)
    z = d * np.sin(b)
    gal_xyz = np.stack((x, y, z)).reshape(3, -1) * u.pc
    schmagal_xyz = gal_to_schmagal(gal_xyz, *gal_args)
    ln_density = ln_prob_density(schmagal_xyz.to_value(u.pc), 0, x_args, y_args, z_args)
    return (ln_density.reshape(d.shape) + 
            2 * np.log(d) + np.log(np.cos(b)))


def get_ln_Veff(x_dens_args, y_dens_args, z_dens_args, gal_args, n_simpson_grid=21):
    all_args = (x_dens_args, y_dens_args, z_dens_args, gal_args)
    
    ranges1 = [
        (0, 2*np.pi),
        (-np.pi/2, -minb.to_value(u.rad)),
        (0, maxdist.to_value(u.pc))
    ]
    ranges2 = [
        (0, 2*np.pi),
        (minb.to_value(u.rad), np.pi/2),
        (0, maxdist.to_value(u.pc))
    ]
    
    log_Veffs = []
    for ranges in [ranges1, ranges2]:
        grids_1d = [np.linspace(*r, n_simpson_grid) for r in ranges]
        grids = np.meshgrid(*grids_1d)
        F = log_integrand(*grids, *all_args)

        xx, yy, zz = grids_1d
        log_Veff = log_simpson(log_simpson([log_simpson(ff, zz) for ff in F], yy), xx)
        log_Veffs.append(log_Veff)
        
    log_Veff = np.logaddexp(*log_Veffs)
    
    return log_Veff