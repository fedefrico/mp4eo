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
Custom Exceptions Submodule
===========================

This submodule provides base classes for exceptions and warnings
when dealing with MP4EO problems.
"""


##############################################################################
#                                  MODULES                                   #
##############################################################################

# Standard library
import warnings

# from qtealeaves.tooling import QTeaLeavesError

##############################################################################
##############################################################################


class BaseMP4EOError(Exception):
    """
    Base custom class for all custom exceptions in the MP4EO library.
    This provides a consistent message formatting and dynamic label handling.

    Parameters
    ----------
    msg : str
        The error message.

    label : str
        The dynamic label to prepend to the message.
    """

    def __init__(self, msg=None, label=None):
        """Constructor."""
        self.msg = msg or "An error occurred in the MP4EO library."
        self.label = label or self.__class__.__name__
        super().__init__(self.formatted_message())

    def formatted_message(self):
        """Generates the formatted error message."""
        return f"[{self.label}] {self.msg}"

    def __str__(self):
        """Return the string representation of the error."""
        return self.formatted_message()


class MP4EOInstanceError(BaseMP4EOError):
    """
    Base custom class for all exceptions related to an MP4EO
    problem instance with a dynamic label.
    """

    def __init__(
        self, message="An error related to an MP4EO problem instance occurred."
    ):
        super().__init__(message, "MP4EOInstanceError")


class MP4EOInstanceConstructionError(BaseMP4EOError):
    """
    Custom exception class for errors in the MP4EO problem
    instance construction.
    """

    def __init__(self, message):
        super().__init__(message, "MP4EOInstanceConstructionError")


class MP4EOInstanceConstructionWarning(UserWarning):
    """
    Custom warning class for MP4EO problem instance
    construction-related warnings.
    """

    def __init__(self, message):
        self.message = f"[MP4EOInstanceConstructionWarning] {message}"
        super().__init__(self.message)

    @classmethod
    def warn(cls, message):
        warnings.warn(cls(message).message, cls, stacklevel=2)


class MP4EOInstanceTypeError(BaseMP4EOError, TypeError):
    """
    Custom exception class for MP4EO problem instance class type error.
    """

    def __init__(self, message):
        super().__init__(message, "MP4EOInstanceTypeError")


class MP4EOSolverError(BaseMP4EOError):
    """
    Base custom class for all exceptions related to an MP4EO
    solver with a dynamic label.
    """

    def __init__(
        self, message="An error related to an MP4EO solver occurred."
    ):
        super().__init__(message, "MP4EOSolverError")


class MP4EOMapperError(BaseMP4EOError):
    """
    Base custom class for all exceptions related to either a
    classical or quantum-inspired mapping process of the
    original MP4EO constrained optimization problem with a
    dynamic label.
    """

    def __init__(
        self,
        message="An error related to a mapping process of an MP4EO instance occurred.",
    ):
        super().__init__(message, "MP4EOMapperError")


class MP4EOQUBOMapperError(BaseMP4EOError):
    """
    Custom exception class for errors during the QUBO mapping
    procedure.
    """

    def __init__(
        self, message="An error related to the QUBO mapping process occurred."
    ):
        super().__init__(message, "QUBOMapperError")


class MP4EOQuditsMapperError(BaseMP4EOError):
    """
    Custom exception class for errors during the qudit mapping
    procedure.
    """

    def __init__(
        self, message="An error related to the qudit mapping process occurred."
    ):
        super().__init__(message, "QuditsMapperError")


class MP4EOQuantumInspiredSolverError(BaseMP4EOError):
    """
    Base custom exception class for all exceptions related to
    a generic quantum-inspired solver of the original MP4EO
    constrained optimization problem with a dynamic label.
    """

    def __init__(
        self,
        message="An error related to a quantum-inspired solver of an MP4EO instance occurred.",
    ):
        super().__init__(message, "QuantumInspiredSolverError")


class MP4EOQuantumInspiredSolverWarning(UserWarning):
    """Custom warning class for MP4EO solver related warnings."""

    def __init__(self, message):
        self.message = f"[MP4EOQuantumInspiredSolverWarning] {message}"
        super().__init__(self.message)

    @classmethod
    def warn(cls, message):
        warnings.warn(cls(message).message, cls, stacklevel=2)


class MP4EOConvParamsError(BaseMP4EOError):
    """
    Custom exception class for errors during the set-up of
    a generic MP4EO solver.
    """

    def __init__(self, message):
        super().__init__(message, "ConvergenceParametersError")


class TensorNetworkSetUpError(BaseMP4EOError):  # , QTeaLeavesError):
    """
    Custom exception class for errors during the set-up of
    the tensor-network solver simulation.
    """

    def __init__(self, message):
        super().__init__(message, "TensorNetworkSetUpError")


class CPLEXSetUpError(BaseMP4EOError):
    """
    Custom exception class for errors during the set-up of
    the CPLEX solver simulation.
    """

    def __init__(self, message):
        super().__init__(message, "CPLEXSetUpError")


class GurobiSetUpError(BaseMP4EOError):
    """
    Custom exception class for errors during the set-up of
    the Gurobi solver simulation.
    """

    def __init__(self, message):
        super().__init__(message, "GurobiSetUpError")


class MP4EOExactSolverError(BaseMP4EOError):
    """
    Custom exception class for errors in the exact quantum
    MP4EO solver.
    """

    def __init__(
        self,
        message="An error related to the exact-diagonalization based solver occurred.",
    ):
        super().__init__(message, "ExactSolverError")


class MP4EOTNSolverError(BaseMP4EOError):
    """
    Custom exception class for errors in the tensor-network
    based MP4EO solver.
    """

    def __init__(
        self,
        message="An error related to the tensor-network based solver occurred.",
    ):
        super().__init__(message, "TNSolverError")
