#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: Songyan Zhu
# Mail: zhusy93@gmail.com
# Created Time:  2018-10-23 13:28:34
#############################################


from setuptools import setup, find_packages

setup(
	name = "uflux",
	version = "0.0.1",
	keywords = ("Geosciences, geospatial, data science"),
	description = "Unified FLUXes (UFLUX)",
	long_description = "Unified FLUXes (UFLUX)",
	license = "MIT Licence",

	url="https://github.com/soonyenju/uflux",
	author = "Songyan Zhu",
	author_email = "zhusy93@gmail.com",

	packages = find_packages(),
	include_package_data = True,
    package_data={
        "uflux.model_parameters": ["*.csv"],
    },
	platforms = "any",
	install_requires=[

	]
)