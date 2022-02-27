# Standard library
import abc

# Third-party
import numpy as np
from scipy.integrate import quad

# This package
from .stats import ln_two_sech2, ln_uniform, ln_exp


class DensityModel(abc.ABC):

    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, 'par_names'):
            cls.par_names = ()

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if key in self.par_names:
                setattr(self, key, val)
            else:
                raise ValueError(f"Unrecognized parameter: {key}")

    @abc.abstractmethod
    def ln_density(self, xyz):
        pass


class UniformSech2DensityModel(DensityModel):
    par_names = (
        'h1',
        'h2',
        'f',
        'x_a',
        'x_b',
        'y_a',
        'y_b'
    )

    def ln_density(self, xyz):
        return (
            ln_uniform(xyz[0], self.x_a, self.x_b) +
            ln_uniform(xyz[1], self.y_a, self.y_b) +
            ln_two_sech2(xyz[2], self.h1, self.h2, self.f)
        )


class ExpSech2DensityModel(DensityModel):
    par_names = (
        'h1',
        'h2',
        'f',
        'R0',
        'h_R'
    )

    def ln_density(self, xyz):
        R = np.sqrt(xyz[0]**2 + xyz[1]**2)
        return (
            ln_exp(R, self.R0, self.h_R) +
            ln_two_sech2(xyz[2], self.h1, self.h2, self.f)
        )


def make_asym_sech2_density_model(basis_funcs):
    coeff_names = [f"a{i}" for i in range(len(basis_funcs))]

    class UniformAsymSech2DensityModel(DensityModel):
        _basis_funcs = basis_funcs
        _coeff_names = tuple(coeff_names)

        par_names = (
            'x_a',
            'x_b',
            'y_a',
            'y_b'
        ) + _coeff_names

        def __init__(self, domain=None, **kwargs):
            super().__init__(**kwargs)

            self._log_normalize = 0
            if domain is not None:
                normalize, err = quad(
                    lambda z: np.exp(self._ln_basis_density(z)),
                    domain[0], domain[1], epsabs=1e-10
                )
                self._log_normalize = -np.log(normalize)

        def _ln_basis_density(self, z):
            dens = 0
            for name, bfunc in zip(self._coeff_names, self._basis_funcs):
                dens = dens + getattr(self, name) * bfunc(z)
            return self._log_normalize + np.log(dens)

        def ln_density(self, xyz):
            return (
                ln_uniform(xyz[0], self.x_a, self.x_b) +
                ln_uniform(xyz[1], self.y_a, self.y_b) +
                self._ln_basis_density(xyz[2])
            )

    return UniformAsymSech2DensityModel
