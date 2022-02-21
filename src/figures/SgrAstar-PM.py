import pathlib
import arviz as az
import corner


this_path = pathlib.Path(__file__).parent
data_path = (this_path / '../data/').resolve()

samples = {}
for name in ['jointNorth', 'jointEast']:
    samples[name] = az.from_netcdf(data_path / f'Reid2020_{name}.netcdf')

data = {
    'pmra': samples['jointEast'].posterior.pm.values.ravel(),
    'pmdec': samples['jointNorth'].posterior.pm.values.ravel(),
    'accra': samples['jointEast'].posterior.acc.values.ravel(),
    'accdec': samples['jointNorth'].posterior.acc.values.ravel()
}

fig = corner.corner(data)
fig.savefig('SgrAstar-PM.pdf')
