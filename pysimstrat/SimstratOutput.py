import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from enum import Enum
from scipy.ndimage.filters import gaussian_filter
from collections import namedtuple
import pysimstrat.date as simdate
from rotseedataset import plottools

import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates


class PlotScale(Enum):
	LINEAR = 0
	LOG = 1


class ModelOutputDefinitions(Enum):
	T = ('T', 'Temperature', '\si{\celsius}', PlotScale.LINEAR)
	nuh = ('nuh', 'Turbulent Diffusivity', '\si{\square\metre\per\second}', PlotScale.LOG)
	P = ('P', 'Shear Stress Production', '\si{\watt\per\kilo\gram}', PlotScale.LOG)
	B = ('B', 'Buoyancy Production', '\si{\watt\per\kilo\gram}', PlotScale.LOG)
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
				self.df = pd.read_csv(os.path.join(self.path, self.shortname + '_out.dat'), header=0)#, delim_whitespace=True)
			except Exception as e:
				error('Could not read model output!')
				print(e)
				sys.exit(1)

			# remove last column that contains the surface values
			#self.df.drop(self.df.columns[-1], axis=1, inplace=True)

			index_col = self.df.columns[0]
			self.df[index_col] = [simdate.days_to_datetime64(date) for date in self.df[index_col]]
			self.df.set_index(index_col, inplace=True)
			self.df.index.name = 'date'
			self.df.columns = [-float(col) for col in self.df.columns]
			#for col in self.df.columns:
			#	self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
			#self.df = self.df.clip(lower=1e-15)

		def as_matrix(self):
			model_dates = self.df.index.values
			model_depths = [float(col) for col in self.df.columns]
			model_val = self.df.as_matrix()
			return (model_dates, model_depths, model_val)

	return ModelOutputClass(*getattr(ModelOutputDefinitions, state_var_shortname).value)


def format_axes(axes):
	for position in ['bottom', 'left', 'right', 'top']:
		axes.spines[position].set_linewidth(1.0)
	axes.get_xaxis().tick_bottom()
	axes.get_yaxis().tick_left()

	# adjust tick position and padding
	axes.get_yaxis().set_tick_params(direction='out', length=5, width=1.0, labelsize=9)
	axes.get_xaxis().set_tick_params(direction='out', length=5, width=1.0, labelsize=9)
	axes.tick_params(axis='both', which='major', pad=5)


def plotModelOutput(modeloutput, levels=2*12, path='./'):
	## read output data
	date, depth, output = modeloutput.as_matrix()
	if modeloutput.plotscale == PlotScale.LOG:
		output = np.log10(output)

	df = pd.DataFrame(data=output, index=date)
	df.columns = depth

	# create figure and axes
	fig = pyplot.figure(figsize=(11.692/3*2,8.267/2))
	ax = fig.add_subplot(111)
	format_axes(ax)
	# format axis
	# label
	ax.set_ylabel(r'Depth [\si{\metre}]')
	ax.invert_yaxis()
	# limits
	# Tick format
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
	#date_locator = mdates.AutoDateLocator()
	#ax.xaxis.set_major_locator(date_locator)
	#fig.autofmt_xdate()

	# create meshgrid and plot contourf
	#date = date.astype(np.float64)
	xx, yy = np.meshgrid(df.index.to_pydatetime(), df.columns)
	cmesh = ax.contourf(xx,yy,np.fliplr(np.rot90(df.as_matrix(), -1)), 40, zorder=-9)#, cmap=cmap)
	pyplot.gca().set_rasterization_zorder(-1)
	# add colorbar
	cbar = fig.colorbar(cmesh, ax=ax, orientation='vertical')
	cbar.ax.get_yaxis().labelpad = 15
	cbar.set_label((r'log10 ' if modeloutput.plotscale==PlotScale.LOG else r'')+modeloutput.name + ' [' + modeloutput.unit + ']', rotation=270)

	fig.tight_layout()
	fig.savefig(os.path.join(path, modeloutput.shortname+'.pdf'), format='pdf', dpi=1200, orientation='landscape', facecolor='white')
	#pyplot.show()


	#fig, ax = plot.createSubplotAxes('SimStrat Rotsee', 1, 1)

	## read output data
	#date, depth, output = modeloutput.as_matrix()
	#if modeloutput.plotscale == PlotScale.LOG:
	#    output = np.log10(output)
	## create plots
	#plot.createProfileTimeSeriesContourf(fig, ax[0],
	#                                      date, depth, output.transpose(),
	#                                      ylabel='Depth [m]', zlabel=('log10 ' if modeloutput.plotscale==PlotScale.LOG else '')+modeloutput.name + ' [' + modeloutput.unit + ']',
	#                                      levels=levels)
	## show plots
	#plt.show()
