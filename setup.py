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
    version="0.0.7",

    author="maojiangyun",
    author_email="maojiangyun@163.com",
    url="https://github.com/ooooona/model_tools.git",
    description="convertor tool for pytorch format to onnx and torchscript",
    long_description=long_description,
    long_description_content_type="text/markdown",


    python_requires='>=3.6',
    entry_points={'console_scripts': [
        'torch2onnx=torch2onnx.torch2onnx:main',
    ]},
    include_package_data=True,
    platforms=['darwin', 'linux'],
    install_requires=[
        "torch",
    ],

    packages=find_packages(),
    license="NU General Public License v3.0",
)
