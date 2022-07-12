#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="drdf",
    version="1.0",
    author="Nilesh Kulkarni",
    description="Code for Scene DRDF",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=[],
)
