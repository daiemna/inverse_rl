# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Daiem Nadir Ali

"""Utility functions."""
from ruamel.yaml import YAML
import logging

logger = logging.getLogger(__name__)


def read_yaml_file(path):
    """Reads a yaml file."""
    with open(path, 'r') as stream:
        yml = YAML(typ='safe')
        return yml.load(stream)