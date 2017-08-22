Python API to generate and manage SimStrat/PEST configurations

Installation
~~~~~~~~~~~~

.. code-block:: python

   pip install pysimstrat


Quick Start
~~~~~

A basic simstrat configuration can be created by providing a lake name, the depth of the lake as well as the start and end date of the simulation (as numpy datetime64):

.. code-block:: python

   import pysimstrat.configuration as simconfig

   simstrat_configuration = simconfig.DefaultSimstratConfiguration(lake_name, lake_depth, startdate, enddate)

The returned object (simstrat_configuration) is a datastructure containing all variables of a full simstrat configuration.
All parameters are initialized with default values but can be adapted to the specific case:

.. code-block:: python

   # change stepsize
   simstrat_configuration.parfile.simulationconfig.stepsize = 300
   # change the stability function that is used (default is 0)
   simstrat_configuration.parfile.modeloption.stabilityfunction = 1
   # change model parameter q_NN
   simstrat_configuration.parfile.modelparameter.q_NN = 0.88

The initial conditions, the morphology as well as the meteorological forcing have no default value and have to be provided as pandas DataFrame:

.. code-block:: python

   simstrat_configuration.morphology.df = ...
   simstrat_configuration.forcing.df = ...
   simstrat_configuration.initialconditions.df = ...

In the same way, the in- and outflow inputfiles can be adapted by providing the concent as a pandas DataFrame (e.g. salinity):

.. code-block:: python
    import pysimstrat.date as simdate
    s_depths = [-8, 0]
    s_source = [0.002, 0]
    salinity_df = pd.DataFrame({'depth': s_depths, simdate.datetime64_to_days(startdate): s_source, simdate.datetime64_to_days(enddate): s_source})
    salinity_df.set_index('depth', inplace=True)
    salinity_df = salinity_df.T
    salinity_df.index.name='date'

    simstrat_configuration.Sinp.df = salinity_df

The complete configuration can be written to files by:

.. code-block:: python
    simstrat_configuration.writeConfiguration()
