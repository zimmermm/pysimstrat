import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from zimjtoolbox.terminalmsg import *
from zimjtoolbox import plothelper as plot
from datetime import datetime
import sys
from enum import Enum
from scipy.ndimage.filters import gaussian_filter
from collections import namedtuple
import pysimstrat.date as simdate

class PlotScale(Enum):
    LINEAR = 0
    LOG = 1


class ModelOutputDefinitions(Enum):
    T = ('T', 'Temperature', 'degC', PlotScale.LINEAR)
    nuh = ('nuh', 'Turbulent Diffusivity', 'm2 d-1', PlotScale.LOG)
    S = ('S', 'Salinity', 'permille', PlotScale.LINEAR)


def getModelOutput(state_var_shortname, path='./Rotsee_Output/'):
    model_output_names = [var.name for var in ModelOutputDefinitions]
    if state_var_shortname is None or state_var_shortname not in model_output_names:
        raise ValueError('Available model outputs are: {0}'.format(' | '.join(model_output_names)))

    class ModelOutputClass(namedtuple('ModelOutput', 'shortname, name, unit, plotscale')):
        def __new__(cls, *args):
            self = super(ModelOutputClass, cls).__new__(cls, *args)
            self.path = path
            self.read()
            return self

        def read(self):
            # try to read the temperature data
            try:
                self.df = pd.read_csv(self.path + self.shortname + '_out.dat', header=0, delim_whitespace=True)
            except Exception as e:
                error('Could not read model output!')
                print(e)
                sys.exit(1)

            # remove last column that contains the surface values
            self.df.drop(self.df.columns[-1], axis=1, inplace=True)

            index_col = self.df.columns[0]
            self.df[index_col] = [simdate.days_to_datetime64(date) for date in self.df[index_col]]
            self.df.set_index(index_col, inplace=True)
            self.df.index.name = 'date'
            self.df.columns = [-float(col) for col in self.df.columns]

        def as_matrix(self):
            model_dates = self.df.index.values
            model_depths = [float(col) for col in self.df.columns]
            model_val = self.df.as_matrix()
            return (model_dates, model_depths, model_val)

    return ModelOutputClass(*getattr(ModelOutputDefinitions, state_var_shortname).value)


def plotModelOutput(modeloutput, levels=2*12):
    # create figure and axes
    fig, ax = plot.createSubplotAxes('SimStrat Rotsee', 1, 1)

    # read output data
    date, depth, output = modeloutput.as_matrix()
    if modeloutput.plotscale == PlotScale.LOG:
        output = np.log10(output)
    # create plots
    plot.createProfileTimeSeriesContourf(fig, ax[0],
                                          date, depth, output.transpose(),
                                          ylabel='Depth [m]', zlabel=('log10 ' if modeloutput.plotscale==PlotScale.LOG else '')+modeloutput.name + ' [' + modeloutput.unit + ']',
                                          title=None, levels=levels)
    # show plots
    plt.show()
