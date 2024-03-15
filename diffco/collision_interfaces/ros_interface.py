# Description: This file contains the interfaces for different robots and environments
# They either read robot info and provides fkine function, or read obstacle info.
# Consider add parent classes for robot and environment interfaces, respectively.

# Copyright (c) Facebook, Inc. and its affiliates.
"""
Differentiable robot model class
====================================
TODO
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os

import torch

class ROSRobot:
    def __init__(self, robot_name):
        raise NotImplementedError
    

