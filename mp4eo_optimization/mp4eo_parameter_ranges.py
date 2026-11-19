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


"""Explicit synthetic parameter interface; no industrial/default ranges.

Public release addition (2026-11-19). Independently written for the synthetic
release under the repository's Apache-2.0 license; no original range values copied.
"""

from numbers import Integral

__all__ = ["get_synthetic_ranges"]

_REQUIRED_KEYS = (
    "N_SAT", "N_TAR", "THETA", "SIGMA", "DTO_TIME", "DLO_TIME",
    "DTO_PREP_TIME", "DLO_PREP_TIME", "MEM_SIZE", "MEM_ONBOARD",
)


def get_synthetic_ranges(parameter_ranges):
    """Validate caller-specified inclusive integer bounds for every required key.

    All ranges must be chosen independently of confidential data. No defaults,
    calibration data, or industrial operating ranges are provided.
    """
    if parameter_ranges is None:
        raise ValueError("Supply independently specified synthetic parameter_ranges.")
    ranges = {}
    for key in _REQUIRED_KEYS:
        if key not in parameter_ranges:
            raise ValueError(f"Missing synthetic range: {key}")
        bounds = tuple(parameter_ranges[key])
        if (len(bounds) != 2 or
                any(not isinstance(v, Integral) or isinstance(v, bool) for v in bounds) or
                bounds[0] > bounds[1]):
            raise ValueError(f"Expected two ordered integer bounds for {key}.")
        ranges[key] = bounds
    return ranges
