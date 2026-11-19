##############################################################################
#                              COPYRIGHT NOTICE                              #
##############################################################################
#
# This code is part of the tesoro_eo_mission_planning_optimization
# project of the INFN Baltig Quantum Padova group repository.
#
# This code is licensed under the Apache License, Version 2.0.
# You may obtain a copy of this license at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must
# retain this copyright notice, and modified files need to
# carry a notice indicating that they have been altered from
# the originals.
#
# The original code is developed by Marco Tesoro.
# The tensor network code is taken from the Quantum TEA
# library developed by the Quantum Padova group.
#
##############################################################################
##############################################################################


"""
Init file for the MP4EO Optimization Module
===========================================

This file initializes the :mod:`mp4eo_optimization`
module, exposing all submodules and their public members
for external use.
The import order respects the dependency tree to avoid
circular imports and ensure correctness. To respect the dependency tree:

    - Submodules that have no dependencies on others should
      be imported first;
    - Submodules that depend on others should be imported only
      after their dependencies.
"""


# Public release modification (2026-11-19): expose core APIs only; experimental solver files are excluded.
# Version management
from .version import __version__

# Core problem and mapping modules.  Keeping these imports independent from the
# solver modules allows the instance model and mappers to be used without the
# experimental tensor-network stack.
from . import mp4eo_instance, mp4eo_mappers, utils
from .exceptions import *
from .mp4eo_instance import *
from .mp4eo_mappers import *
from .utils import *

__all__ = (
    utils.__all__
    + mp4eo_instance.__all__
    + mp4eo_mappers.__all__
)
