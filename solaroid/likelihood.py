import astropy.units as u
from gala.units import UnitSystem
import numpy as np

from .coordinates import gal_to_schmagal
from .integrate import log_simpson


class Model:

    def __init__(self, data_gal_xyz, DensityModel,
                 sgrA_star, min_abs_b, max_dist, usys,
                 frozen=None):
        self.usys = UnitSystem(usys)

        self.min_abs_b = min_abs_b
        self._min_abs_b = self.min_abs_b.to_value(u.rad)

        self.max_dist = max_dist
        self._max_dist = self.max_dist.to_value(self.usys['length'])

        self.data_gal_xyz = data_gal_xyz
        self.sgrA_star = sgrA_star

        self.DensityModel = DensityModel

        self.par_names = (
            'ln_n0',
            'zsun',
            'roll'
        ) + self.DensityModel.par_names

        if frozen is None:
            frozen = {}
        self.frozen = dict(frozen)

    def ln_integrand(self, l, b, d, density_model, gal_args):
        l, b, d = map(np.array, [l, b, d])

        x = d * np.cos(l) * np.cos(b)
        y = d * np.sin(l) * np.cos(b)
        z = d * np.sin(b)
        gal_xyz = np.stack((x, y, z)).reshape(3, -1) * u.pc
        schmagal_xyz = gal_to_schmagal(gal_xyz, *gal_args)

        val = density_model.ln_density(
            schmagal_xyz.to_value(self.usys['length'])
        )
        return (
            val.reshape(d.shape) +
            2 * np.log(d) + np.log(np.cos(b))
        )

    def get_ln_Veff(self, density_model, gal_args, n_simpson_grid=21):
        ranges1 = [
            (0, 2*np.pi),
            (-np.pi/2, -self._min_abs_b),
            (0, self._max_dist)
        ]
        ranges2 = [
            (0, 2*np.pi),
            (self._min_abs_b, np.pi/2),
            (0, self._max_dist)
        ]

        log_Veffs = []
        for ranges in [ranges1, ranges2]:
            grids_1d = [np.linspace(*r, n_simpson_grid) for r in ranges]
            grids = np.meshgrid(*grids_1d)
            F = self.ln_integrand(*grids, density_model, gal_args)

            xx, yy, zz = grids_1d
            log_Veff = log_simpson(
                log_simpson(
                    [log_simpson(ff, zz) for ff in F],
                    yy
                ),
                xx
            )
            log_Veffs.append(log_Veff)

        log_Veff = np.logaddexp(*log_Veffs)

        return log_Veff

    def unpack_pars(self, p_arr):
        p_dict = self.frozen.copy()

        i = 0
        for name in self.par_names:
            if name not in self.frozen:
                p_dict[name] = p_arr[i]
                i += 1

        return p_dict

    def pack_pars(self, p_dict):
        p = []
        for name in self.par_names:
            if name not in self.frozen:
                val = p_dict[name]
                if hasattr(val, 'unit'):
                    val = val.decompose(self.usys).value
                p.append(val)
        return np.array(p)

    def _get_density_model(self, p_dict, fill_frozen=True):
        p_dict = p_dict.copy()

        if fill_frozen:
            p_dict.update(self.frozen)

        for k, v in p_dict.items():
            if hasattr(p_dict[k], 'unit'):
                p_dict[k] = v.decompose(self.usys).value

        kw = {
            k: v for k, v in p_dict.items()
            if k in self.DensityModel.par_names
        }

        density_model = self.DensityModel(**kw)
        return density_model

    def ln_likelihood(self, p, plot=False):
        pars = self.unpack_pars(p)

        gal_args = (
            self.sgrA_star,
            pars['zsun'] * self.usys['length'],
            pars['roll'] * self.usys['angle']
        )
        rot_xyz = gal_to_schmagal(self.data_gal_xyz, *gal_args)
        rot_xyz = rot_xyz.to_value(self.usys['length'])

        density_model = self._get_density_model(pars)

        if plot:
            from .stats import ln_two_sech2
            import matplotlib.pyplot as plt
            grid = np.linspace(-5000, 5000, 128)
            plt.hist(rot_xyz[2], bins=grid, density=True)

            val = np.exp(ln_two_sech2(grid,
                                      h1=pars['h1'],
                                      h2=pars['h2'],
                                      f=pars['f']))
            plt.plot(grid, val, marker='')
            plt.yscale('log')

        ln_Veff = self.get_ln_Veff(density_model, gal_args)

        return (
            - np.exp(pars['ln_n0'] + ln_Veff) +
            rot_xyz.shape[1] * pars['ln_n0'] +
            density_model.ln_density(rot_xyz).sum() +
            np.log(rot_xyz.shape[1])
        )
