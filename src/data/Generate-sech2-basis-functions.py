# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

# +
import copy
import pathlib
import yaml

import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simpson, quad

from solaroid.simulate import invt_sample_z
from solaroid.likelihood import ln_normal, ln_two_sech2
from solaroid.integrate import log_simpson
from solaroid.gram_schmidt import gram_schmidt
# -

_path = pathlib.Path('../src/static/')
with open(_path / 'fiducial-density.yml', 'r') as f:
    pars = yaml.safe_load(f.read())
pars


def inner_product(f1, f2, measure_func, scale):
    return quad(lambda z: f1(z) * f2(z) * measure_func(scale * z) * scale, -np.inf, np.inf)[0]


# +
funcs = [np.polynomial.Polynomial.basis(deg) for deg in range(15)]

measure_func = lambda z: np.exp(ln_two_sech2(z, **pars))
grid = np.arctanh(np.linspace(-1+1e-8, 1-1e-8, 8192))

sech2_basis_funcs = gram_schmidt(funcs, inner_product, args=(measure_func, pars['h2']))
# -

sech2_basis_funcs[8]

plot_grid = np.linspace(-10, 10, 1024)
for func in sech2_basis_funcs:
    plt.plot(plot_grid, func(plot_grid), marker='')
# plt.xlim(-5000, 5000)
plt.xlim(-10, 10)
plt.ylim(-5, 5)

plot_grid = np.linspace(-10, 10, 1024)
for func in sech2_basis_funcs[:8]:
    plt.plot(
        plot_grid * pars['h2'] / 1e3, 
        func(plot_grid) * measure_func(plot_grid * pars['h2']), 
        marker=''
    )
# plt.xlim(-5000, 5000)
plt.xlim(-3, 3)
plt.xlabel('$z$ [kpc]')


