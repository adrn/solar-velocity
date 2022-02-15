import astropy.units as u
import numpy as np
from coord_helpers import gal_to_schmagal, schmagal_to_gal
from integrate_helpers import log_simpson


def ln_normal(x, mu, std):
    return -0.5 * (x-mu)**2 / std**2 - 0.5*np.log(2*np.pi) - np.log(std)


def ln_uniform(x, a, b):
    return -np.log(b - a)


def ln_two_sech2(z, h1, h2, f):
    lnterm1 = np.log(f) - 2 * np.log(np.cosh(z / (2 * h1))) - np.log(4 * h1)
    lnterm2 = np.log(1 - f) - 2 * np.log(np.cosh(z / (2 * h2))) - np.log(4 * h2)
    return np.logaddexp(lnterm1, lnterm2)


def ln_density(
    xyz, 
    ln_N0, 
    ln_x_prob, x_args, 
    ln_y_prob, y_args,
    ln_z_prob, z_args
):
    return (
        ln_N0 + 
        ln_x_prob(xyz[0], *x_args) + 
        ln_y_prob(xyz[1], *y_args) + 
        ln_z_prob(xyz[2], *z_args)
    )


def ln_integrand(l, b, d, ln_density_args, gal_args):
    l, b, d = map(np.array, [l, b, d])
    
    x = d * np.cos(l) * np.cos(b)
    y = d * np.sin(l) * np.cos(b)
    z = d * np.sin(b)
    gal_xyz = np.stack((x, y, z)).reshape(3, -1) * u.pc
    schmagal_xyz = gal_to_schmagal(gal_xyz, *gal_args)
    val = ln_density(schmagal_xyz.to_value(u.pc), 0, *ln_density_args)
    return (
        val.reshape(d.shape) + 
        2 * np.log(d) + np.log(np.cos(b))
    )


def get_ln_Veff(ln_density_args, gal_args, min_abs_b, max_dist, n_simpson_grid=21):
    ranges1 = [
        (0, 2*np.pi),
        (-np.pi/2, -min_abs_b.to_value(u.rad)),
        (0, max_dist.to_value(u.pc))
    ]
    ranges2 = [
        (0, 2*np.pi),
        (min_abs_b.to_value(u.rad), np.pi/2),
        (0, max_dist.to_value(u.pc))
    ]
    
    log_Veffs = []
    for ranges in [ranges1, ranges2]:
        grids_1d = [np.linspace(*r, n_simpson_grid) for r in ranges]
        grids = np.meshgrid(*grids_1d)
        F = ln_integrand(*grids, ln_density_args, gal_args)

        xx, yy, zz = grids_1d
        log_Veff = log_simpson(log_simpson([log_simpson(ff, zz) for ff in F], yy), xx)
        log_Veffs.append(log_Veff)
        
    log_Veff = np.logaddexp(*log_Veffs)
    
    return log_Veff


def ln_likelihood(p, xyz, sgrA_star, ln_density_args, min_abs_b, max_dist, plot=False):
    lnn0, lnh1, lnh2, f, zsun, roll = p
    
    gal_args = (sgrA_star, zsun * u.pc, roll * u.rad)
    rot_xyz = gal_to_schmagal(xyz * u.pc, *gal_args).to_value(u.pc)
    
    z_args = (np.exp(lnh1), np.exp(lnh2), f)
    
    if plot:
        import matplotlib.pyplot as plt
        grid = np.linspace(-5000, 5000, 128)
        plt.hist(rot_xyz[2], bins=grid, density=True);

        val = np.exp(ln_two_sech2(grid, *z_args))
        plt.plot(grid, val, marker='')
        plt.yscale('log')
    
    args = (
        ln_density_args[0], ln_density_args[1], # TODO: could swap out for params?
        ln_density_args[2], ln_density_args[3], # TODO: could swap out for params?
        ln_density_args[4], z_args
    )
    ln_Veff = get_ln_Veff(args, gal_args, min_abs_b, max_dist)
    
    return (
        - np.exp(lnn0 + ln_Veff) + 
        xyz.shape[1] * lnn0 +
        ln_density(rot_xyz, 0, *args).sum() +
        np.log(xyz.shape[1])
    )
