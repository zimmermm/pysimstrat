import numpy as np
import pandas as pd

def generateForcingDataFrame(meteo_data):
    if meteo_data is None or not isinstance(meteo_data, pd.DataFrame):
        raise ValueError('MeteoData have to be provided as pandas DataFrame')
    else:
        ws = meteo_data['wind_scalar']
        angle = meteo_data['wind_direction']
        u = np.multiply(ws, np.sin(np.divide(angle, 360. / (2. * np.pi))))/3.6
        v = np.multiply(ws, np.cos(np.divide(angle, 360. / (2. * np.pi))))/3.6
        SimStratForcing = pd.DataFrame(data={'t': meteo_data['date'],
                                             'u (m/s)': pd.Series(u),
                                             'v (m/s)': pd.Series(v),
                                             'Tair (' + u"\N{DEGREE SIGN}" + 'C)': pd.Series(meteo_data['air_temperature']),
                                             'Fsol (W/m2)': pd.Series(meteo_data['global_radiation']),
                                             'vap (mbar)': pd.Series(meteo_data['vapour_pressure']),
                                             'cloud coverage': pd.Series(1.-meteo_data['sunshine']/100.)})
        SimStratForcing = SimStratForcing[['t', 'u (m/s)', 'v (m/s)', 'Tair (' + u"\N{DEGREE SIGN}" + 'C)', 'Fsol (W/m2)', 'vap (mbar)', 'cloud coverage']]
        return SimStratForcing
