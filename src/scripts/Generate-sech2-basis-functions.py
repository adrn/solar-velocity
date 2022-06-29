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
import os
import pathlib
import pickle
import yaml
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from scipy.integrate import quad

from solaroid.stats import ln_two_sech2
from solaroid.gram_schmidt import gram_schmidt
import paths

# +

_path = paths.static
with open(_path / 'fiducial-density.yml', 'r') as f:
    pars = yaml.safe_load(f.read())
print(f"Fiducial parameters: {pars}")


# -

def inner_product(f1, f2, measure_func, scale):
    return quad(lambda z: f1(z) * f2(z) * measure_func(scale * z) * scale,
                -np.inf, np.inf)[0]


# +
MAXORDER = 25
funcs = [np.polynomial.Polynomial.basis(deg) for deg in range(MAXORDER)]


def measure_func(z):
    return np.exp(ln_two_sech2(z, **pars))


grid = np.arctanh(np.linspace(-1+1e-8, 1-1e-8, 8192))

sech2_basis_funcs = gram_schmidt(funcs, inner_product,
                                 args=(measure_func, pars['h2']))

for func in sech2_basis_funcs:
    print(func)

with open(paths.data / "basis-funcs.pkl", "wb") as f:
    pickle.dump(sech2_basis_funcs, f)
# -

plot_grid = np.linspace(-10, 10, 1024)
scale = pars['h2'] / 1e3

fig, ax = plt.subplots()
for func in sech2_basis_funcs:
    ax.plot(plot_grid * scale, func(plot_grid), marker='')
ax.set_xlim(-4, 4)
ax.set_ylim(-5, 5)
ax.set_xlabel('$z$ [kpc]')
fig.tight_layout()
fig.savefig(paths.data / 'basis-funcs.pdf')

fig, ax = plt.subplots()
for func in sech2_basis_funcs:
    ax.plot(
        plot_grid * scale,
        func(plot_grid) * measure_func(plot_grid * pars['h2']),
        marker=''
    )
ax.set_xlim(-3, 3)
ax.set_xlabel('$z$ [kpc]')
fig.tight_layout()
fig.savefig(paths.data / 'basis-funcs-measure.pdf')

# +
fig, ax = plt.subplots(figsize=(6, 10))

past_max = 0
past_offset = 0
fudge = 0
for i, func in enumerate(sech2_basis_funcs):
    func_vals = func(plot_grid) * measure_func(plot_grid * pars['h2'])
    if i > 0:
        offset = past_max + fudge + np.abs(min(func_vals)) + past_offset
        plot_vals = func_vals + offset
    else:
        plot_vals = func_vals
        offset = 0
        fudge = 0.1 * max(func_vals)

    ax.plot(
        plot_grid * scale,
        plot_vals,
        marker=''
    )
    ax.plot(
        func.roots() * pars['h2'] / 1e3,
        offset * np.ones_like(func.roots()),
        marker='o',
        ls='none'
    )
    past_max = max(func_vals)
    past_offset = offset

ax.set_xlim(-3, 3)
ax.set_xlabel('$z$ [kpc]')
fig.tight_layout()

# +
rng = np.random.default_rng(seed=42)

vals = 0
for coeff, func in zip(rng.uniform(size=16), sech2_basis_funcs):
    vals = vals + coeff * func(plot_grid) * measure_func(plot_grid * pars['h2'])

fig, ax = plt.subplots()
ax.plot(
    plot_grid * scale,
    vals,
    marker=''
)
ax.set_xlim(-3, 3)
ax.set_xlabel('$z$ [kpc]')
fig.tight_layout()
