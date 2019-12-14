from distutils.core import setup
from glob import glob
from os.path import splitext, basename
from setuptools import find_packages

setup(name='Meepo',
      version='1.0',
      description='Joined attitude detection algorithm',
      author='Maksim Zayakin, Mark Averchenko, Vladislav Zaripov',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
     )