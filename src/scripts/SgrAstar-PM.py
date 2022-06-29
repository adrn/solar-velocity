import pathlib
import arviz as az
import astropy.table as at
import astropy.units as u
import corner
import numpy as np
import paths

# this_path = pathlib.Path(__file__).parent
data_path = paths.data

samples = {}
for name in ['jointNorth', 'jointEast']:
    samples[name] = az.from_netcdf(data_path / f'Reid2020_{name}.netcdf')

data = {
    'pmra': samples['jointEast'].posterior.pm.values.ravel(),
    'pmdec': samples['jointNorth'].posterior.pm.values.ravel(),
    'accra': samples['jointEast'].posterior.acc.values.ravel() * 1e3,
    'accdec': samples['jointNorth'].posterior.acc.values.ravel() * 1e3
}

fit_vals = at.QTable.read('../data/Reid2020_refit.ecsv')[0]

fig = corner.corner(
    data,
    plot_density=False,
    plot_datapoints=False,
    contour_kwargs=dict(alpha=1, colors='tab:blue', linewidths=[1, 2]),
    color='tab:blue',
    levels=[1 - np.exp(-val**2 / 2) for val in [1, 2]],
    bins=31,
    range=[
        (fit_vals['pmra'].value - 0.1, fit_vals['pmra'].value + 0.1),
        (fit_vals['pmdec'].value - 0.1, fit_vals['pmdec'].value + 0.1),
        (-8, 8),
        (-8, 8)
    ],
    labels=[
        r"$\mu_\alpha$ " + f"[{u.mas/u.yr:latex_inline}]",
        r"$\mu_\delta$ " + f"[{u.mas/u.yr:latex_inline}]",
        r"$\mu_\alpha$ " + f"[{u.nanoarcsecond/u.yr**2:latex_inline}]",
        r"$\mu_\delta$ " + f"[{u.nanoarcsecond/u.yr**2:latex_inline}]"
    ],
    label_kwargs=dict(fontsize=16),
    labelpad=0.1
)
fig.suptitle("Sgr A* astrometry (fit to data from Reid & Brunthaler 2020)",
             fontsize=18, y=0.99)
fig.subplots_adjust(bottom=0.12, left=0.12)
fig.savefig('SgrAstar-PM.pdf')


