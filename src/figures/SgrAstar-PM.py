import pathlib
import arviz as az


this_path = pathlib.Path(__file__).parent
data_path = (this_path / '../data/').resolve()

samples = {}
for name in ['jointNorth', 'jointEast']:
    samples[name] = az.from_netcdf(data_path / f'Reid2020_{name}.netcdf')

print(samples)
