import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'iga'
AUTHOR = 'Vitalis Vosylius'
LICENSE = 'MIT'


INSTALL_REQUIRES = []

setup(name=PACKAGE_NAME,
      version=VERSION,
      author=AUTHOR,
      license=LICENSE,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
