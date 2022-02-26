# Standard library
import abc

# Third-party
import numpy as np


class DensityModel:

    def __init__(self, ln_amp):
        self.ln_amp = ln_amp

    @abc.abstractmethod
    def ln_density(self, xyz):
        pass


class CartesianDensityModel(DensityModel):

    def __init__(
        self,
        ln_amp,
        x_func, y_func, z_func,
        x_args=(), y_args=(), z_args=()
    ):
        super().__init__(ln_amp)
        self.x_func = lambda x: x_func(x, *x_args)
        self.y_func = lambda x: y_func(x, *y_args)
        self.z_func = lambda x: z_func(x, *z_args)

    def ln_density(self, xyz):
        return (
            self.ln_amp +
            self.x_func(xyz[0]) +
            self.y_func(xyz[1]) +
            self.z_func(xyz[2])
        )


class DiskDensityModel(DensityModel):

    def __init__(
        self,
        ln_amp,
        R_func, z_func,
        R_args=(), z_args=()
    ):
        super().__init__(ln_amp)
        self.R_func = lambda x: R_func(x, *R_args)
        self.z_func = lambda x: z_func(x, *z_args)

    def ln_density(self, xyz):
        return (
            self.ln_amp +
            self.R_func(np.sqrt(xyz[0]**2 + xyz[1]**2)) +
            self.z_func(xyz[2])
        )
