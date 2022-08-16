from setuptools import find_packages
from setuptools import setup

setup(
    name='bivariate',
    version='2.0.0',
    description='This package contains methods that assist in performing\
        bivariate analysis of datasets.',
    url='https://gitlab.tudelft.nl/mude/bivariate_package.git',
    packages=find_packages(exclude=['old_package*']),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib >= 3.5.0',
        'seaborn',
        'scipy',
        'ipykernel'
    ]  
)