# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
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

from jupytext.config import find_jupytext_configuration_file
find_jupytext_configuration_file('.')

# +
import pathlib

import astropy.coordinates as coord
import astropy.table as at
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import theano.tensor as tt

import pymc3 as pm
import pymc3_ext as pmx
import corner
# -

rng = np.random.default_rng(seed=42)

# See: Table 1 caption in Reid & Brunthaler 2020
fiducial_c = coord.SkyCoord(
    "17:45:40.0409",
    "-29:00:28.118",
    unit=(u.hourangle, u.degree)
)

# The galactic center coordinate in Galactocentric is taken from the Appendix of Reid & Brunthaler 2004. That is from the definition of the origin in Galactic coordinates, propagated to J2000/ICRS frame. In the future, we should use the location of Sgr A* at an epoch. Here we adopt 2016 as the epoch (Gaia DR3).

(fiducial_c.ra.degree,
 fiducial_c.dec.degree)

(coord.Galactocentric().galcen_coord.ra.degree,
 coord.Galactocentric().galcen_coord.dec.degree)

diff_sep = fiducial_c.separation(coord.Galactocentric().galcen_coord)
(diff_sep * 8.2*u.kpc).to(u.pc, u.dimensionless_angles())

# ---

this_path = pathlib.Path(__file__).parent
static_path = (this_path / '../static/').resolve()
data_path = (this_path / '../data/').resolve()

data = {}
for filename in static_path.glob('J*'):
    name = filename.parts[-1]
    tbl = at.QTable.read(filename, format='ascii.csv')
    tbl['Date'] = Time(tbl['Date'], format='jyear')
    for colname in tbl.colnames[1:]:
        tbl[colname] *= u.mas
    tbl['dEast'] = -tbl['dEast']
    tbl['dNorth'] = -tbl['dNorth']
    data[name] = tbl

# +
style = dict(ls='none', marker='o', ms=2)

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5),
    sharex=True,
    constrained_layout=True
)

for name, tbl in data.items():
    print(np.min(tbl['dEast_err']), np.min(tbl['dNorth_err']))
    axes[0].errorbar(
        tbl['Date'].jyear,
        tbl['dEast'].value,
        tbl['dEast_err'].value,
        **style
    )

    axes[1].errorbar(
        tbl['Date'].jyear,
        tbl['dNorth'].value,
        tbl['dNorth_err'].value,
        **style
    )

for ax in axes:
    ax.set_xlabel('year')
axes[0].set_ylabel(r'$\Delta\alpha$ [mas]')
axes[1].set_ylabel(r'$\Delta\delta$ [mas]')

# +
EPOCH = 2000.


def make_model(t_jyear, dx, dx_err):
    with pm.Model() as model:
        acc = pm.Uniform('acc', -10, 10)  # acceleration in mas/yr**2
        pm_ = pm.Uniform('pm', -10, 10)  # proper motion in mas/yr
        x0 = pm.Uniform('x0', -1000, 1000)
        logs = pm.Uniform('logs', -12, 2)
        s = tt.exp(logs)
        err = tt.sqrt(s**2 + dx_err**2)
        true_dx = acc * (t_jyear - EPOCH)**2 + pm_ * (t_jyear - EPOCH) + x0
        pm.Normal('like', true_dx, err, observed=dx)

    return model


def make_joint_model(t_jyear, dx, dx_err, ids):
    with pm.Model() as model:
        acc = pm.Uniform('acc', -10, 10)  # acceleration in mas/yr**2
        pm_ = pm.Uniform('pm', -10, 10)  # proper motion in mas/yr

        for id_ in np.unique(ids):
            mask = ids == id_
            x = t_jyear[mask]
            y = dx[mask]
            y_err = dx_err[mask]

            x0 = pm.Uniform(f'x0_{id_}', -1000, 1000)
            logs = pm.Uniform(f'logs_{id_}', -12, 2)
            s = tt.exp(logs)
            err = tt.sqrt(s**2 + y_err**2)
            true_dx = acc * (x - EPOCH)**2 + pm_ * (x - EPOCH) + x0
            pm.Normal(f'like_{id_}', true_dx, err, observed=y)

    return model


# -

seed = int(rng.integers(0, 100_000))
sample_kw = dict(
    tune=1000, draws=10000,
    chains=2,
    cores=1,
    return_inferencedata=True,
    random_seed=seed
)

all_samples = {}
for name, tbl in data.items():
    for dir_ in ['East', 'North']:
        with make_model(tbl['Date'].jyear, tbl[f'd{dir_}'].value,
                        tbl[f'd{dir_}_err'].value) as model:
            res = pmx.optimize(start={'pm': -3, 'x0': 0})
            print(res)
            all_samples[name + dir_] = pmx.sample(start=res, **sample_kw)

# +
# Joint fit:
tbl = at.vstack((data['J1745-283'], data['J1748-291']))
tbl['id'] = np.ones(len(tbl), dtype=int)
tbl['id'][len(data['J1745-283']):] = 2

for dir_ in ['East', 'North']:
    with make_joint_model(tbl['Date'].jyear, tbl[f'd{dir_}'].value,
                          tbl[f'd{dir_}_err'].value, tbl['id']) as model:
        res = pmx.optimize(start={'pm': -3, 'x0': 0})
        print(res)
        all_samples['joint' + dir_] = pmx.sample(start=res, **sample_kw)

# -

pm.summary(all_samples['jointEast'])

pm.summary(all_samples['jointNorth'])

# +
pm_east = np.mean(all_samples['jointEast'].posterior.pm.values.ravel())
pm_east_err = np.std(all_samples['jointEast'].posterior.pm.values.ravel())

pm_north = np.mean(all_samples['jointNorth'].posterior.pm.values.ravel())
pm_north_err = np.std(all_samples['jointNorth'].posterior.pm.values.ravel())
# -

m = np.stack((all_samples['jointEast'].posterior.pm.values.ravel(),
              all_samples['jointNorth'].posterior.pm.values.ravel()))
np.cov(m)

print(f"pm_E = {pm_east:.3f} +/- {pm_east_err:.3f}")
print(f"pm_N = {pm_north:.3f} +/- {pm_north_err:.3f}")

pos_east_2016 = (
    all_samples['jointEast'].posterior.pm * (2016 - EPOCH)
    + all_samples['jointEast'].posterior.x0_1
)
pos_north_2016 = (
    all_samples['jointNorth'].posterior.pm * (2016 - EPOCH)
    + all_samples['jointNorth'].posterior.x0_1
)

np.mean(pos_east_2016).values, np.mean(pos_north_2016).values

np.std(pos_east_2016).values, np.std(pos_north_2016).values

sgr_ra_2016 = fiducial_c.ra + np.mean(pos_east_2016).values * u.mas
sgr_dec_2016 = fiducial_c.dec + np.mean(pos_north_2016).values * u.mas

# +
Rsun = 8.275 * u.kpc
cc = coord.SkyCoord(
    sgr_ra_2016,
    sgr_dec_2016,
    distance=Rsun,
    pm_ra_cosdec=pm_east * u.mas/u.yr,
    pm_dec=pm_north * u.mas/u.yr,
    radial_velocity=0*u.km/u.s
)

galcen_frame = coord.Galactocentric(
    galcen_v_sun=[0, 0, 0]*u.km/u.s,
    galcen_distance=Rsun,
    z_sun=20.8 * u.pc,
    galcen_coord=coord.SkyCoord(sgr_ra_2016, sgr_dec_2016)
)
# -

-cc.transform_to(galcen_frame).velocity.d_xyz

all_samples.keys()

colors = ['tab:blue', 'tab:orange', 'k']

fig = None
for name, color in zip(['J1748-291East', 'J1745-283East', 'jointEast'],
                       colors):
    if fig is None:
        fig = corner.corner(all_samples[name].posterior, color=color,
                            var_names=['pm', 'acc'])
    else:
        fig = corner.corner(all_samples[name].posterior, fig=fig, color=color,
                            var_names=['pm', 'acc'])

fig = None
for name, color in zip(['J1748-291North', 'J1745-283North', 'jointNorth'],
                       colors):
    if fig is None:
        fig = corner.corner(all_samples[name].posterior, color=color,
                            var_names=['pm', 'acc'])
    else:
        fig = corner.corner(all_samples[name].posterior, fig=fig, color=color,
                            var_names=['pm', 'acc'])

results = {
    'epoch': 2016,
    'ra': sgr_ra_2016,
    'ra_err': np.std(pos_east_2016).values * u.mas,
    'dec': sgr_dec_2016,
    'dec_err': np.std(pos_north_2016).values * u.mas,
    'pmra': pm_east * u.mas/u.yr,
    'pmra_err': pm_east_err * u.mas/u.yr,
    'pmdec': pm_north * u.mas/u.yr,
    'pmdec_err': pm_north_err * u.mas/u.yr,
}
results = at.QTable([results])
results.write(data_path / 'Reid2020_refit.ecsv',
              overwrite=True)
results


