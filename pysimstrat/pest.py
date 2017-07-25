import sys, os
import pandas as pd
import numpy as np
import pysimstrat.configuration as simconfig
import pysimstrat.date as simdate
from collections import namedtuple, OrderedDict
from namedlist import namedlist
import itertools


##################################################
# Parameter Definition
##################################################

ParameterDefinitionType = namedlist('ParameterDefinition', 'PARNME, PARTRANS, PARCHGLIM, PARVAL1, PARLBND, PARUBND, PARGP, SCALE, OFFSET, DERCOM')


class ParameterDefinition(ParameterDefinitionType):
    def fit(self):
        self.PARTRANS = 'none'
        self.PARGP = 'fit'
        return self

    def fix(self):
        self.PARTRANS = 'fixed'
        self.PARGP = 'none'
        return self

    def relative(self):
        self.PARCHLIM = 'relative'
        return self

    def factor(self):
        self.PARCHLIM = 'factor'
        return self

    def log(self):
        self.PARTRANS = 'log'
        return self

    def lin(self):
        self.PARTRANS = 'nonw'
        return self


def ParameterDefinitionFactory(name, val, upper, lower):
    return ParameterDefinition(name, 'fixed', 'factor', val, upper, lower, 'none', 1.0, 0.0, 1)


def ParameterDefinitions(parameterdefinitions):
    labels = ', '.join([definition.PARNME for definition in parameterdefinitions])
    ParameterDefinitionsType = namedlist('ParameterDefinitions', labels)

    class ParameterDefinitions(ParameterDefinitionsType):
        def __str__(self):
            return '\n'.join(['\t'.join([str(getattr(definition, field)).ljust(8) for field in definition._fields]) for definition in self])

        def __iter__(self):
            return iter([getattr(self, field) for field in self._fields])

    return ParameterDefinitions(*parameterdefinitions)


DefaultParfileParameterDefinitions = ParameterDefinitions([ParameterDefinitionFactory('lat', 47.55, 40.0, 50.0),
                                                           ParameterDefinitionFactory('p_air', 990, 900, 1000),
                                                           ParameterDefinitionFactory('a_seiche', 0.01, 0.001, 0.01),
                                                           ParameterDefinitionFactory('q_NN', 1.0, 0.7, 1.25),
                                                           ParameterDefinitionFactory('f_wind', 1.0, 0.1, 2.0),
                                                           ParameterDefinitionFactory('C10', 0.0015, 0.001, 0.003),
                                                           ParameterDefinitionFactory('CD', 0.002, 0.001, 0.005),
                                                           ParameterDefinitionFactory('fgeo', 0, 0, 1.0).relative(),
                                                           ParameterDefinitionFactory('k_min', 1e-30, 1e-30, 1e-9),
                                                           ParameterDefinitionFactory('p1', 1.0, 0.8, 1.2),
                                                           ParameterDefinitionFactory('p2', 1.0, 0.9, 1.1),
                                                           ParameterDefinitionFactory('beta', 0.35, 0, 0.5).relative(),
                                                           ParameterDefinitionFactory('albsw', 0.08, 0, 0.3).relative(),
                                                          ])


##################################################
# Inputfile Mask
##################################################


def flag(name, marker='#', maxlen=10):
    return marker+name+' '*(maxlen-len(name))+marker


class InputfileMask(object):
    def __init__(self, inputclass, masked_data):
        self.inputclass = inputclass
        self.masked_data = masked_data

    def cast(self, inputfile, masked_data, marker):
        if self.inputclass.__name__ is 'InputList':
            setattr(inputfile, inputfile._fields[1], masked_data)
        elif self.inputclass.__name__ is 'InputTabular':
            inputfile.df = masked_data


class ParfileMask(InputfileMask):
    def __init__(self):
        pass

    def cast(self, parfile, masked_data, marker):
        model_parameter_names = simconfig.Parfile.SECTIONS['modelparameter'].section_class._fields
        for param in model_parameter_names:
            setattr(parfile.modelparameter, param, flag(param))


def createInputfileMask(inputclass, masked_data):
    if inputclass is simconfig.Parfile:
        return ParfileMask()
    else:
        return InputfileMask(inputclass, masked_data)


##################################################
# Template
##################################################


TemplateDefinition = namedtuple('Templatedefinition', 'inputfile, masked_data, parameterdefinitions')
def generateTemplate(templatedefinition, marker):
    # unpack templatedefinition
    filename = templatedefinition.inputfile.filename
    inputfile_class = templatedefinition.inputfile.__class__
    masked_data = templatedefinition.masked_data
    parameterdefinitions = templatedefinition.parameterdefinitions
    if inputfile_class is simconfig.Parfile and parameterdefinitions is None:
        parameterdefinitions = DefaultParfileParameterDefinitions

    # class definition of an inputfile template
    class InputfileTemplate(inputfile_class):
        def __init__(self, filename, mask):
            templatefilename = ''.join([filename.rsplit('.', 1)[0],'.tpl'])
            self.mask = mask
            super(InputfileTemplate, self).__init__(filename)
            super(InputfileTemplate, self).read()
            self.inputfile_name = self.filename
            self.filename = templatefilename
            self.mask.cast(self, masked_data, marker)
            self.parameterdefinitions = parameterdefinitions

        def write(self):
            super(InputfileTemplate, self).write()
            with open(self.filename, 'r', encoding='utf-8') as templatefile:
                content = templatefile.read()

            with open(self.filename, 'w', encoding='utf-8') as templatefile:
                pest_header = 'ptf ' + marker
                templatefile.write('\n'.join([pest_header, content]))

    return InputfileTemplate(filename, createInputfileMask(inputfile_class, masked_data))


##################################################
# Output Instruction
##################################################


OutputInstructionDefinition = namedlist('OutputInstructionDefinition', [('observation_df', None), ('outputfilepath', None), ('float_format', '{:.2f}')])
OutputDefinition = namedtuple('OutputDefinition', 'outputdepths, outputtimes')
def generateOutputInstructions(outputinstructiondefinitions):
    # get unique observation depths and dates
    outputdepths = pd.concat([outputinstruction.observation_df['depth'] for outputinstruction in outputinstructiondefinitions]).unique()
    outputtimestamps = pd.concat([outputinstruction.observation_df['date'] for outputinstruction in outputinstructiondefinitions]).unique()
    # sort
    outputdepths.sort()
    outputdepths = outputdepths[::-1]
    outputtimestamps.sort()
    outputtimestamps = np.apply_along_axis(simdate.datetime64_to_days, 0, outputtimestamps)
    # create instructions
    outputdefinition = OutputDefinition(outputdepths, outputtimestamps)
    return [OutputInstruction(outputinstructiondefinition, outputdefinition) for outputinstructiondefinition in outputinstructiondefinitions]


class OutputInstruction(object):
    def __init__(self, outputinstructiondefinition, outputdefinition):
        # make all fields available
        for field in outputinstructiondefinition._fields:
            setattr(self, field, getattr(outputinstructiondefinition, field))
        for field in outputdefinition._fields:
            setattr(self, field, getattr(outputdefinition, field))

        # translate filename
        self.filename = os.path.split(self.outputfilepath)[1].rsplit('.', 1)[0]+'_output.ins'

        # extract some general info
        if self.observation_df.columns[0] is not 'date' and self.observation_df.columns[1] is not 'depth':
            raise ValueError('The observation dataframe columns do not follow the required pattern.')

        obs_label = self.observation_df.columns[2]
        n_obs = self.observation_df.shape[0]

        # generate observation data
        self.observationdata = pd.DataFrame(OrderedDict([
            ('OBSNME', [obs_label+str(i) for i in range(n_obs)]),
            ('OBSVAL', self.observation_df[obs_label].map(outputinstructiondefinition.float_format.format)),
            ('WEIGHT', np.ones((n_obs, 1)).ravel()),
            ('OBGNME', 'obs'),
        ]))

    def write(self):
        with open(self.filename, 'w', encoding='utf-8') as outputinstruction:
            header = 'pif %'
            depths = self.outputdepths[::-1]
            dates = self.outputtimes
            instructions_df = self.observation_df.copy()
            instructions_df['OBSNME'] = self.observationdata['OBSNME'].copy()

            def createInstructions():
                for date in dates:
                    instruction_df = instructions_df.loc[(instructions_df['date']==simdate.days_to_datetime64(date))]
                    marker = '%{:.4f}%'.format(date)
                    idx = [np.where(depths==instruction_depth)[0][0] for instruction_depth in instruction_df['depth']]
                    instructions = ['w']*(max(idx)+2)
                    for i, nme in zip(idx, instruction_df['OBSNME']):
                        instructions[i+1] = '!{0}!'.format(nme)
                    yield ' '.join([marker]+instructions)

            outputinstruction.write('\n'.join([header]+list(createInstructions())))


class ObservationData(object):
    def __init__(self, outputinstructions):
        self.outputdata = pd.concat([outputinstruction.observationdata for outputinstruction in outputinstructions])

    def __iter__(self):
        return self.outputdata.iterrows()

    def __len__(self):
        return self.outputdata.shape[0]

    def __str__(self):
        #output_df = self.outputdata.copy()
        #output_df['OBSVAL'] = output_df['OBSVAL'].map('{:.2f}'.format)
        return self.outputdata.to_csv(sep='\t', index=False, header=False).rstrip()


##################################################
# Control File
##################################################


class ModelInputOutput(object):
    def __init__(self, templates, outputinstructions):
        self.templates = templates
        self.outputinstructions = outputinstructions

    def __str__(self):
        return '\n'.join(['\n'.join([template.filename + '\t' + template.inputfile_name for template in self.templates]),
                          '\n'.join([outputinstruction.filename + '\t' + outputinstruction.outputfilepath for outputinstruction in self.outputinstructions])
                         ])


PESTControlFileType = namedlist('PESTControlFile',
                                [
                                    ('RSTFLE', 'restart'),
                                    ('PESTMODE', 'estimation'),
                                    ('NPAR', None),
                                    ('NOBS', None),
                                    ('NPARGP', 1),
                                    ('NPRIOR', 0),
                                    ('NOBSGP', 1),
                                    ('NTPLFLE', None),
                                    ('NINSFLE', None),
                                    ('PRECIS', 'double'),
                                    ('DPOINT', 'nopoint'),
                                    ('NUMCOM', 1),
                                    ('JACFILE', 0),
                                    ('MESSFILE', 0),
                                    ('RLAMBDA1', 5.0),
                                    ('RLAMFAC', 2.0),
                                    ('PHIRATSUF', 0.3),
                                    ('PHIREDLAM', 0.01),
                                    ('NUMLAM', 10),
                                    ('RELPARMAX', 1.0),
                                    ('FACPARMAX', 1.5),
                                    ('FACORIG', 0.001),
                                    ('PHIREDSWH', 0.1),
                                    ('NOPTMAX', 30),
                                    ('PHIREDSTP', 0.005),
                                    ('NPHISTP', 4),
                                    ('NPHINORED', 3),
                                    ('RELPARSTP', 0.01),
                                    ('NRELPAR', 3),
                                    ('ICOV', 1),
                                    ('ICOR', 1),
                                    ('IEIG', 1),
                                    ('PARGPNME', 'fit'),
                                    ('INCTYPE', 'relative'),
                                    ('DERINC', 0.01),
                                    ('DERINCLB', 0.00001),
                                    ('FORCEN', 'switch'),
                                    ('DERINCMUL', 2.0),
                                    ('DERMTHD', 'parabolic'),
                                    ('parameterdata', None),
                                    ('OBGNME', 'obs'),
                                    ('observationdata', None),
                                    ('MCLI', None),
                                    ('modelinout', None),
                                    ('priorinfo', '')
                                ])


class PESTControlFile(PESTControlFileType):
    SECTION = namedtuple('PESTControlFileSection', 'header, keys')
    STRUCTURE = [SECTION('* control data', (
                                                ('RSTFLE', 'PESTMODE'),
                                                ('NPAR', 'NOBS', 'NPARGP', 'NPRIOR', 'NOBSGP'),
                                                ('NTPLFLE', 'NINSFLE', 'PRECIS', 'DPOINT', 'NUMCOM', 'JACFILE', 'MESSFILE'),
                                                ('RLAMBDA1', 'RLAMFAC', 'PHIRATSUF', 'PHIREDLAM', 'NUMLAM'),
                                                ('RELPARMAX', 'FACPARMAX', 'FACORIG'),
                                                ('PHIREDSWH',),
                                                ('NOPTMAX', 'PHIREDSTP', 'NPHISTP', 'NPHINORED', 'RELPARSTP', 'NRELPAR'),
                                                ('ICOV', 'ICOR', 'IEIG')
                                           )),
                SECTION('* parameter groups', (
                                                ('PARGPNME', 'INCTYPE', 'DERINC', 'DERINCLB', 'FORCEN', 'DERINCMUL', 'DERMTHD'),
                                           )),
                SECTION('* parameter data', (
                                                ('parameterdata',),
                                           )),
                SECTION('* observation groups', (
                                                ('OBGNME',),
                                           )),
                SECTION('* observation data', (
                                                ('observationdata',),
                                           )),
                SECTION('* model command line', (
                                                ('MCLI',),
                                           )),
                SECTION('* model input/output', (
                                                ('modelinout',),
                                           )),
                SECTION('* prior information', (
                                                ('priorinfo',),
                                           ))
                ]

    def __init__(self, filename, templates, parameterdefinitions, outputinstructions, modelcommand):
        super(PESTControlFile, self).__init__()
        self.filename = filename
        self.parameterdata = parameterdefinitions
        self.outputinstructions = outputinstructions
        self.observationdata = ObservationData(self.outputinstructions)
        self.MCLI = modelcommand
        self.NPAR = len(self.parameterdata)
        self.NOBS = len(self.observationdata)
        self.NTPLFLE = len(templates)
        self.NINSFLE = len(self.outputinstructions)
        self.modelinout = ModelInputOutput(templates, self.outputinstructions)

    def write(self):
        with open(self.filename, 'w', encoding='utf-8') as pestcontrolfile:
            pestcontrolfile.write('\n'.join(['pcf']+
                                            ['\n'.join([section.header,
                                                        '\n'.join(
                                                                    ['\t'.join([str(getattr(self, key)) for key in line])
                                                                    for line in section.keys]
                                                                 )
                                                       ])
                                            for section in PESTControlFile.STRUCTURE])
                                 )


class SimstratPESTConfiguration(object):
    def __init__(self, simstratconfiguration, templatedefinitions, outputinstructiondefinitions, modelcommand, marker='#'):
        self.marker = marker
        self.simstratconfiguration = simstratconfiguration
        self.templatedefinitions = templatedefinitions

        self.templates = [generateTemplate(templatedefinition, marker) for templatedefinition in templatedefinitions]
        self.parameterdefinitions = ParameterDefinitions(list(itertools.chain.from_iterable([template.parameterdefinitions for template in self.templates])))
        self.outputinstructions = generateOutputInstructions(outputinstructiondefinitions)
        self.controlfile = PESTControlFile(simstratconfiguration.parfile.filename.rsplit('.', 1)[0]+'_ctr.pst', self.templates, self.parameterdefinitions, self.outputinstructions, modelcommand)

    def write(self):
        for template in self.templates:
            template.write()

        for outputinstruction in self.outputinstructions:
            outputinstruction.write()

        self.controlfile.write()

        # write output format instructions
        self.simstratconfiguration.outputdepths.depths = self.outputinstructions[0].outputdepths.tolist()
        self.simstratconfiguration.outputtime.time = self.outputinstructions[0].outputtimes.tolist()
        self.simstratconfiguration.writeConfiguration()


#def generatePESTTemperatureOutputInstructions(inputfile, obs_dates, obs_depths, obs_names):
#    # assert that the files exist that are needed to generate the instruction file
#    assert os.path.exists(inputfile['input_paths'][GRID])
#    assert os.path.exists(inputfile['input_paths'][OUTPUT_TIMEINTERVAL])
#
#    # read the output depths
#    output_depths = -pd.read_csv(inputfile['input_paths'][OUTPUT_DEPTH])['depth'].values[::-1]
#
#    print('create temperature output instruction file...')
#    instruction_file_path = inputfile['input_file_name'].replace('.par', '_obs.ins')
#    header = 'pif %'
#
#    def create_instruction_line(obs_pos, date):
#        # first maker is the timestamp
#        marker = '%{:.4f}%'.format(datetime2float(date)+0.5)
#        # all observation depths for the given timestamp
#        depths = obs_depths[np.where(obs_dates == date)]
#        depths_idx = [int(i) for i in np.array([np.where(output_depths == depth)[0] for depth in depths]).ravel()]
#        obs_nme = obs_names[obs_pos:obs_pos+len(depths_idx)]
#        flags = ['w']*(max(depths_idx)+2)
#        for idx, nme in zip(depths_idx, obs_nme):
#            flags[idx+1] = '!'+nme+'!'
#        return (obs_pos+len(depths_idx), ' '.join([marker]+flags))
#
#    def map_special(f, recursion, arglist):
#        for arg in arglist:
#            recursion, result = f(recursion, arg)
#            yield result
#
#    instruction_lines = list(map_special(create_instruction_line, 0, np.unique(obs_dates)))
#
#    with open(instruction_file_path, 'w') as instruction_file:
#        instruction_file.write('\n'.join([header]+instruction_lines))
#
#    print('check instruction file...')
#    subprocess.call(['E:/rootworkspace/pest13/inschek.exe', instruction_file_path])
#
#
