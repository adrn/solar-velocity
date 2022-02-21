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

import pathlib
import astropy.table as at
import astropy.units as u
# %matplotlib inline
import numpy as np

# See: Tables 2 and 3
# https://www.aanda.org/articles/aa/pdf/2021/03/aa40208-20.pdf

gravity = {
    'distance': 8.275 * u.kpc,
    'distance_err': np.sqrt(9**2 + 33**2) * u.pc,
    'radial_velocity': (-2.6 + 11.1) * u.km/u.s,
    'radial_velocity_err': 1.4 * u.km/u.s
}
gravity_tbl = at.QTable([gravity])

this_path = pathlib.Path(__file__).parent
reid_tbl = at.QTable.read(this_path / 'Reid2020_refit.ecsv')

sgrA = at.hstack((reid_tbl, gravity_tbl))
sgrA.write(this_path / 'sgrA_star.ecsv', overwrite=True)
