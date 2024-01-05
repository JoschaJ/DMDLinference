#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Set up some paths for file storage.

Following https://medium.com/@dataproducts/python-the-pythonic-way-to-handle-file-directories-for-your-data-project-6b19417463ad
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PACKAGE_DIR, 'Data')
DATA_IN_DIR = os.path.join(PACKAGE_DIR, 'DMDLinference')