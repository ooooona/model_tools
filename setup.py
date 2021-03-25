#!/usr/bin/env python
# -*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: oona
# Mail: maojiangyun@163.com
# Created Time:  2032-03-25 19:00:59
#############################################


from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="torch2onnx",
    version="0.0.1",

    author="maojiangyun",
    author_email="maojiangyun@163.com",
    description="convertor tool for pytorch format to onnx and torchscript",
    long_description=long_description,
    long_description_content_type="text/markdown",

    license="NU General Public License v3.0",

    url="https://github.com/ooooona/model_tools.git",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        "torch",
    ]
)
