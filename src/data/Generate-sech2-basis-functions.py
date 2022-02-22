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
import pathlib
import pickle
import yaml
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from scipy.integrate import quad

from solaroid.likelihood import ln_two_sech2
from solaroid.gram_schmidt import gram_schmidt
# -

this_path = pathlib.Path(__file__).parent
_path = (this_path / '../static').resolve()
with open(_path / 'fiducial-density.yml', 'r') as f:
    pars = yaml.safe_load(f.read())
print(f"Fiducial parameters: {pars}")


def inner_product(f1, f2, measure_func, scale):
    return quad(lambda z: f1(z) * f2(z) * measure_func(scale * z) * scale,
                -np.inf, np.inf)[0]


# +
MAXORDER = 15
funcs = [np.polynomial.Polynomial.basis(deg) for deg in range(MAXORDER)]
for func in funcs:
    print(func)

with open(this_path / "basis-funcs.pkl", "wb") as f:
    pickle.dump(funcs, f)


def measure_func(z):
    return np.exp(ln_two_sech2(z, **pars))


grid = np.arctanh(np.linspace(-1+1e-8, 1-1e-8, 8192))

sech2_basis_funcs = gram_schmidt(funcs, inner_product,
                                 args=(measure_func, pars['h2']))
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
fig.savefig(this_path / 'basis-funcs.pdf')

fig, ax = plt.subplots()

for func in sech2_basis_funcs[:8]:
    ax.plot(
        plot_grid * scale,
        func(plot_grid) * measure_func(plot_grid * pars['h2']),
        marker=''
    )
ax.set_xlim(-3, 3)
ax.set_xlabel('$z$ [kpc]')
fig.tight_layout()
fig.savefig(this_path / 'basis-funcs-measure.pdf')
