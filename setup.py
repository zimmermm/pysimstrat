from setuptools import setup

setup(name='pysimstrat',
      version='1.0.0',
      description='python library to manage simstrat & PEST configurations',
      author='Matthias Zimmermann',
      author_email='matthias.zimmermann@eawag.ch',
      url='https://github.com/zimmermm/pysimstrat',
      packages=['pysimstrat'],
      install_requires=['numpy>=1.13', 'scipy>=0.19', 'pandas>=0.20', 'namedlist', 'matplotlib'],
      zip_safe=False)
