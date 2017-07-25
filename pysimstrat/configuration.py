# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime
import os
from collections import OrderedDict, namedtuple
from namedlist import namedlist
from enum import Enum
from itertools import islice
from pysimstrat.date import *


''' DATA STRUCTURE DEFINITION '''
''' ************************* '''

ParfileType = namedlist('Parfile', 'inputfiles, outputfolder, simulationconfig, modeloptions, modelparameter', default=None)
class Parfile(ParfileType):
    Section = namedtuple('Section', 'header, section_class')
    SECTIONS = OrderedDict([
                            ('inputfiles', Section('*** Files *************************************************',
                                                   namedlist('InputFiles', 'initialconditions, grid, morphology, forcing, absorption, outputfolder, outputdepths, outputtime, Qinp, Qout, Tinp, Sinp'))
                            ),
                            ('simulationconfig', Section('*** Model set up: time steps and grid resolution ***',
                                                        namedlist('SimulationConfig', [('stepsize', 300), ('startdate', 0), ('enddate', 1)]))
                            ),
                            ('modeloptions', Section('*** Model, conditions and output selection ****************',
                                                     namedlist('ModelOptions', [('turbulencemodel', 1), ('stabilityfunction', 2), ('fluxcondition', 1), ('forcingtype', 3), ('usefilteredwind', 0), ('seichenormalization', 2), ('winddragmodel', 3), ('inflowplacement', 0), ('pressuregradient', 0), ('enablesalinitytransport', 1), ('displaysimulation', 0), ('displaydiagnose', 1), ('averagedate', 10)]))
                            ),
                            ('modelparameter', Section('*** Model parameters **************************************',
                                                       namedlist('ModelParameter', [('lat', 47.55), ('p_air', 990), ('a_seiche', 0.001), ('q_NN', 1), ('f_wind', 1), ('C10', 1), ('CD', 0.002), ('fgeo', 0.1), ('k_min', 1e-30), ('p1', 1), ('p2', 1), ('beta', 0.35), ('albsw', 0.08)]))
                            )
                          ])
    COMMENTS = {'stepsize': 'Timestep dt [s]',
                'startdate': 'Start time [d]',
                'enddate': 'End time [d]',

                'turbulencemodel': 'Turbulence model (1:k-epsilon, 2:MY)',
                'stabilityfunction': 'Stability function (1:constant, 2:quasi-equilibrium)',
                'fluxcondition': 'Flux condition (0:Dirichlet condition, 1:no-flux)',
                'forcingtype': 'Forcing (1:Wind+Temp+SolRad, 2:(1)+Vap, 3:(2)+Cloud, 4:Wind+HeatFlux+SolRad)',
                'usefilteredwind': 'Use filtered wind to compute seiche energy (0/default:off, 1:on) (if 1:on, one more column is needed in forcing file)',
                'seichenormalization': 'Seiche normalization (1:max N^2, 2:integral)',
                'winddragmodel': 'Wind drag model (1/default:lazy (constant), 2:ocean (increasing), 3:lake (Wüest and Lorke 2003))',
                'inflowplacement': 'Inflow placement (0/default:manual, 1:density-driven)',
                'pressuregradient': 'Pressure gradients (0:off, 1:Svensson 1978, 2:?)',
                'enablesalinitytransport': 'Enable salinity transport (0:off, 1/default:on)',
                'displaysimulation': 'Display simulation (0:off, 1:when data is saved, 2:at each iteration, 3:extra display)',
                'displaydiagnose': 'Display diagnose (0:off, 1:standard display, 2:extra display)',
                'averagedate': 'Averaging data (not implemented, set to 10)',

                'lat': 'Lat [°]         Latitude for Coriolis parameter',
                'p_air': 'p_air [mbar]	Air pressure',
                'a_seiche': 'a_seiche [-]	Fraction of wind energy to seiche energy',
                'q_NN': 'q_NN			Fit parameter for distribution of seiche energy',
                'f_wind': 'f_wind	[-]		Fraction of forcing wind to wind at 10m (W10/Wf)',
                'C10': 'C10 [-]			Wind drag coefficient (used if wind drag model is 1:lazy)',
                'CD': 'CD [-]			Bottom friction coefficient',
                'fgeo': 'fgeo [W/m2]		Geothermal heat flux',
                'k_min': 'k_min [J/kg]	Minimal value for TKE',
                'p1': 'p1				Fit parameter for absorption of IR radiation from sky',
                'p2': 'p2				Fit parameter for convective and latent heat fluxes',
                'beta': 'beta [-]		Fraction of short-wave radiation directly absorbed as heat',
                'albsw': 'albsw [-]		Albedo for reflection of short-wave radiation'}

    def __init__(self, filename):
        super(Parfile, self).__init__()
        self.filename = filename

    def float_if_float(self, string):
        try:
            return float(string)
        except ValueError:
            return string

    def read(self):
        assert os.path.isfile(self.filename)
        with open(self.filename, 'r', encoding='utf-8') as parfile:
            lines = parfile.read().split('\n')
            # remove section headers (lines starting with '*') and empty lines
            lines = [line for line in lines if line is not '' and line[0] is not '*']
            # extract the instructions: fist word as instruction, rest of the line is comment
            instructions = iter([self.float_if_float(line.split(None, 1)[0]) for line in lines])
            # inputfile instructions
            for section in Parfile.SECTIONS:
                section_class = Parfile.SECTIONS[section].section_class
                setattr(self, section, section_class(*list(islice(instructions, len(section_class._fields)))))

        # check validity
        outputfolder_idx = Parfile.SECTIONS['inputfiles'].section_class._fields.index('outputfolder')
        assert all([os.path.isfile(f) for i, f in enumerate(self.inputfiles) if i != outputfolder_idx])

    def write(self):
        def generate_section(section_header, section):
            section_dict = section._asdict()
            ordered_params = section._fields
            # [section_header] + [param    comment]
            return '\n'.join([section_header]+
                             ['\t'.join([str(section_dict[param]).ljust(8), Parfile.COMMENTS[param]]) if param in Parfile.COMMENTS.keys() else str(section_dict[param]).ljust(8) for param in ordered_params]
                            )

        with open(self.filename, 'w', encoding='utf-8') as parfile:
            parfile.write('\n'.join([generate_section(Parfile.SECTIONS[section].header, getattr(self, section)) for section in Parfile.SECTIONS]))


def simstrat_outputfoldertype():
    class OutputFolder():
        def __init__(self, filename, content=None):
            pass

        def read(self):
            pass

        def write(self):
            pass
    return OutputFolder


def simstrat_listtype(columns, float_formats, parname):
    SimstratListType = namedlist('ListType', 'columns, float_formats, '+parname, default=None)
    class InputList(SimstratListType):
        def __init__(self, filename, content=None):
            super(InputList, self).__init__()
            self.filename = filename
            self.columns = list(columns)
            self.float_formats = list(float_formats)
            setattr(self, parname, content)

        def read(self):
            df = pd.read_csv(self.filename, sep='\t', header=0)
            if len(self.columns) > 1:
                df.columns = self.columns
                setattr(self, parname, df)
            else:
                setattr(self, parname, list(df.values.ravel()))

        def write(self):
            data = getattr(self, parname)
            if data is None:
                with open(self.filename, 'w', encoding='utf-8') as inputfile:
                    inputfile.write('\t'.join(self.columns))
            elif isinstance(data, pd.DataFrame):
                data.columns = self.columns
                for col, float_format in zip(data, self.float_formats):
                    def format_col(value):
                        return value if type(value) is str else float_format.format(value)
                    data[col] = data[col].map(format_col)
                data.to_csv(self.filename, sep='\t', index=False, encoding='utf-8')
            elif type(data) is list:
                with open(self.filename, 'w', encoding='utf-8') as inputfile:
                    float_format = 0 if len(data) > 1 else 1
                    lines = '\n'.join([self.columns[0]] + [self.float_formats[float_format].format(item) for item in data])
                    inputfile.write(lines)
            else:
                raise ValueError(self.filename + ': Data must be 1d list or pandas DataFrame')
    return InputList


def simstrat_tabulartype(index_name, header_name, parname):
    SimstratTabularType = namedlist('TabularType', 'index_name, header_name, parname, df', default=None)
    class InputTabular(SimstratTabularType):
        def __init__(self, filename, content=None):
            super(InputTabular, self).__init__(index_name, header_name, parname)
            self.filename = filename
            self.df = content

        def read(self):
            self.df = pd.read_csv(self.filename, sep='\t', skiprows=2, header=0, index_col=0)

        def write(self):
            if self.df is None:
                with open(self.filename, 'w', encoding='utf-8') as inputfile:
                    inputfile.write('\t'.join([self.index_name, self.header_name, self.parname]))
            else:
                self.df.index.name = '-1'
                self.df.to_csv(self.filename, sep='\t', encoding='utf-8')
                with open(self.filename, 'r', encoding='utf-8') as inputfile:
                    content = inputfile.read()
                with open(self.filename, 'w', encoding='utf-8') as inputfile:
                    content = '\n'.join(['\t'.join([self.index_name, self.header_name, self.parname])]+[str(self.df.shape[1])]+[content])
                    inputfile.write(content)

    return InputTabular


class SimstratInputFileDefinitions(Enum):
    grid = simstrat_listtype(['number/depths of gridpoints'], ['{:.2f}']*2, 'gridpoints')
    outputtime = simstrat_listtype(['timepoints or nr of time steps'], ['{:.6f}', '{:.2f}'], 'time')
    outputdepths = simstrat_listtype(['depths'], ['{:.2f}', '{:.2f}'], 'depths')
    morphology = simstrat_listtype(['depth [m]', 'area [m2]'], ['{:.2f}', '{:.0f}'], 'df')
    initialconditions = simstrat_listtype(['depth (m)', 'u (m/s)', 'v (m/s)', 'T (' + u"\N{DEGREE SIGN}" + 'C)', 'S', 'k', 'eps'], ['{:.2f}']*5+['{:.1E}']*2, 'df')
    Qinp, Qout, Tinp, Sinp = list(simstrat_tabulartype(*definition) for definition in list(['t(1. column)', 'z_inp(1. row)', parname] for parname in ['Qinp', 'Qout', 'Tinp', 'Sinp']))
    absorption = simstrat_tabulartype('t', 'z_ga (1. row)', 'ga1')
    forcing = simstrat_listtype(['t', 'u (m/s)', 'v (m/s)', 'Tair (' + u"\N{DEGREE SIGN}" + 'C)', 'Fsol (W/m2)', 'vap (mbar)', 'cloud coverage'], ['{:.6f}']*7, 'df')
    outputfolder = simstrat_outputfoldertype()


''' API                       '''
''' ************************* '''


class SimstratConfiguration(object):
    def __init__(self, parfile_name):
        assert os.path.isfile(parfile_name)
        self.parfile_name = parfile_name

        self.readConfiguration()

    def __iter__(self):
        return iter(getattr(self, inputfile.name) for inputfile in SimstratInputFileDefinitions)

    def readConfiguration(self):
        self.parfile = Parfile(self.parfile_name)
        self.parfile.read()

        inputfile_paths = self.parfile.inputfiles._asdict()
        inputfile_classes = SimstratInputFileDefinitions.__members__

        for inputfile in inputfile_paths:
            setattr(self, inputfile, inputfile_classes[inputfile].value(inputfile_paths[inputfile]))
            getattr(self, inputfile).read()

    def writeConfiguration(self):
        self.parfile.write()
        for inputfile in self:
            inputfile.write()


class DefaultSimstratConfiguration(SimstratConfiguration):
    def __init__(self, lakename, depth, startdate, enddate, absorption=0.2):
        # generate filenames
        lakefileprefix      = ''.join(lakename.split()).lower()
        lakefolderprefix = ''.join(lakename.split()).title()
        inputfolder    = lakefolderprefix+'_Input/'
        outputfolder = lakefolderprefix+'_Output/'
        inputprefix         = inputfolder+lakefileprefix+'_'
        inputfiles = Parfile.SECTIONS['inputfiles'].section_class._fields
        outputfolder_idx = Parfile.SECTIONS['inputfiles'].section_class._fields.index('outputfolder')
        inputfile_paths = list(inputprefix+inputfile+'.dat' for inputfile in inputfiles)
        inputfile_paths[outputfolder_idx] = outputfolder

        # generate folders if needed
        if not os.path.isdir(inputfolder):
            os.mkdir(inputfolder)
        if not os.path.isdir(outputfolder):
            os.mkdir(outputfolder)

        # generate parfile
        self.parfile_name = lakefileprefix+'_simstrat.par'
        self.parfile = Parfile(self.parfile_name)
        # fill with default parameter
        for section, fields in zip(Parfile.SECTIONS, [inputfile_paths, None, None, None]):
            if fields is None:
                setattr(self.parfile, section, Parfile.SECTIONS[section].section_class())
            else:
                setattr(self.parfile, section, Parfile.SECTIONS[section].section_class(*fields))

        # convert startdate and enddate if needed
        if type(startdate) is not np.datetime64:
            startdate = np.datetime64(pd.to_datetime(startdate), 'ns')

        if type(enddate) is not np.datetime64:
            enddate = np.datetime64(pd.to_datetime(enddate), 'ns')

        # set start- and enddate
        self.parfile.simulationconfig.startdate = datetime64_to_days(startdate)
        self.parfile.simulationconfig.enddate = datetime64_to_days(enddate)

        # generate the default inputfile contents
        DEFAULTS = self.generate_defaults(depth, self.parfile.simulationconfig.startdate, self.parfile.simulationconfig.enddate, absorption)

        # generate the inputfiles
        inputfile_paths = self.parfile.inputfiles._asdict()
        inputfile_classes = SimstratInputFileDefinitions.__members__
        for inputfile in inputfile_paths:
            setattr(self, inputfile, inputfile_classes[inputfile].value(inputfile_paths[inputfile], content=DEFAULTS[inputfile]))

    def generate_defaults(self, depth, startdate, enddate, absorption):
        lakedefaults = {'morphology': pd.DataFrame({'depth [m]': [0], 'area [m2]': [0]}),
                        'grid': [800.0],
                        'outputtime': [48],
                        'outputdepths': None,
                        'forcing': None,
                        'initialconditions': None,
                        'absorption': pd.DataFrame({'-0.1': [absorption, absorption]}, index=[startdate, enddate]),
                        'outputfolder': None}
        inflowdefaults = dict([(parname, pd.DataFrame({str(-depth): [0, 0], '0': [0, 0]}, index=[startdate, enddate], )) for parname in ['Qinp', 'Qout', 'Tinp', 'Sinp']])

        combined_dict = lakedefaults.copy()
        combined_dict.update(inflowdefaults)
        return combined_dict
