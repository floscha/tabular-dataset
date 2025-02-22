from setuptools import find_packages
from setuptools import setup


setup(
    name='Tabular Dataset',
    version='0.0',
    url='https://github.com/floscha/tabular-dataset',
    author='Florian Schäfer',
    author_email='florian.joh.schaefer@gmail.com',
    packages=find_packages(exclude=('tests',)),
    install_requires=open('requirements.txt').readlines()
)
