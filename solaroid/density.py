# Standard library
import abc

# Third-party
import numpy as np

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
