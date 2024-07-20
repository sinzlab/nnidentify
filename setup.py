#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='nnidentify',
    version='0.0',
    description='Neural System Identification with Deep Neural Networks',
    author='Paweł A. Pierzchlewicz',
    author_email='ppierzc@gmail.com',
    packages=find_packages(exclude=[]),
    install_requires=[],
)
