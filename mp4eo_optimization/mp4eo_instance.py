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
MP4EO Instance Submodule
========================

This submodule provides tools to instantiate, manage and analyze
a Mission Planning for Earth Observation problem (MP4EO) instance.
"""


##############################################################################
#                                  MODULES                                   #
##############################################################################

# Standard library
from collections import Counter
from functools import cache
import json
from math import floor, log2
from pathlib import Path
import pickle
import re

# Third-party libraries
import numpy as np

# First-party libraries
from mp4eo_optimization.exceptions import (
    MP4EOInstanceConstructionError,
    MP4EOInstanceConstructionWarning,
)
from mp4eo_optimization.mp4eo_parameter_ranges import get_synthetic_ranges
from mp4eo_optimization.utils import ensure_init

__all__ = ["MP4EOInstance"]

##############################################################################
##############################################################################


# Public release modification (2026-11-19): exclude industrial adapters/defaults; accept explicit synthetic ranges.
class MP4EOInstance:
    r"""
    An instance of a Mission Planning for Earth Observation
    problem.

    Parameters
    ----------
    n_satellites : int
        The number of satellites in the constellation.

    n_targets : int
        The number of acquisition requests (targets)
        from the user.
        These are area on the Earth surface to be acquired.

    theta : np.ndarray[int]
        The set of numbers of possible opportunities
        :math:`\vartheta_{ij}` in which satellite
        :math:`i` can acquire a desidered target :math:`j`.

    n_dto : int
        The total number of Data Take Opportunities (DTO) decision
        variables.

    dto_times : np.ndarray[float]
        The set of instantaneous times :math:`t_{ij}^k` in
        which satellite :math:`i` acquire target :math:`j` at
        the opportunity :math:`k`.

    dto_preparation_times : np.ndarray[float]
        The set of preparation times :math:`p_{ij}^k` of
        satellite :math:`i` to gain target :math:`j` at
        occurrence :math:`k`.

    sigma : np.ndarray[int]
        The set of numbers of possibile opportunities
        :math:`\sigma_i` in which satellite :math:`i` can
        perform the downlink of an acquired target.

    n_dlo : int
        The total number of DownLink Opportunities (DLO) decision
        variables.

    dlo_times : np.ndarray[float]
        The set of instantaneous times :math:`s_{ij}^r` in
        which satellite :math:`i` performes the download of
        target :math:`j` at the opportunity :math:`r`.

    dlo_preparation_times : np.ndarray[float]
        The set of preparation downlink times :math:`d_{ij}^r`
        of satellite :math:`i` to download target :math:`j` at
        occurence :math:`r`.

    required_target_memories : np.ndarray[float]
        The set of needed memories :math:`q_{ij}` to gain target
        :math:`j` from satellite :math:`i`.

    total_available_memories : np.ndarray[float]
        The set of total memories :math:`q_i` available from
        satellite :math:`i`.

    alpha : float
        The weight :math:`\alpha` associated to the DTO part
        of the cost function.

    gamma : float
        The weight :math:`\gamma` associated to the DTO part
        of the cost function.

    beta : float
        The weight :math:`\beta` associated to the DLO part
        of the cost function.

    omega_dto : np.ndarray[float]
        The set of DTO-binary weights in the cost function
        :math:`\omega^{\mathrm{DTO}}_{ijk}`.

    omega_dlo : np.ndarray[float]
        The set of DLO-binary weights in the cost function
        :math:`\omega^{\mathrm{DLO}}_{ijr}`.

    iparam_range : Dict
        Ranges of the MP4EO parameters to generate a random
        instance or validate parameters from an input file.

    binaries_arrangement : str, optional
        The order in which binary decision variables are
        arranged within the bitstring representing a generic
        planned mission.
        Available arrangements are as follows:

        - **satellite-wise** arranges all variables associated
          with each individual satellite in ascending target order,
          starting with the DTOs followed by the DLOs, where the
          satellite index is the slowest-varying parameter;
        - **target-wise** introduces all the binary variables associated
          with the same target, with the satellite index incrementing in
          ascending order, starting with the DTOs followed by the DLOs;
        - **type-wise** introduces all the DTOs variables first, with the
          target index varying more slowly than the satellite one;

        Default to ``target-wise``, i.e., the target index is the slowest varying one.

    constraints : Dict, optional
        Provides information about the constraints to be
        considered during optimization. Each key represents
        the name of a specific constraint ``C*``, while the
        corresponding value is a tuple. The first element of
        the tuple is a boolean flag indicating whether the
        constraint should be included. The second element is
        a float that specifies the weight of the constraint
        within the cost function.
        Default to :obj:`None`, i.e., free optimization.

    Attributes
    ----------
    _N : int
        The number of satellites in the constellation.

    _M : int
        The number of targets.

    _n_slack_vars : int
        The number of slack variables :math:`s_i` to enforce
        constraints.

    _binaries_per_slack : np.ndarray[int]
        The number of binaries :math:`b_{i, \alpha}` to represent
        each slack variable :math:`s_i`.

    _binaries_arrangement : str
        The chosen arrangement of the problem decision variables.

    _dto_bin_sg_index_map : Dict[tuple, int]
        Pre-computed dictionary to store the mapping from
        DTO binary indices to the corresponding position
        along a one-dimensional arrangement.

    _dlo_bin_sg_index_map : Dict[tuple, int]
        Pre-computed dictionary to store the mapping from
        DLO binary indices to the corresponding position
        along a one-dimensional arrangement.

    _slack_bin_sg_index_map : Dict[tuple, int]
        Pre-computed dictionary to store the mapping from
        SLACK binary indices to the corresponding position
        along a one-dimensional arrangement.

    iparam_ranges : Dict
        Ranges of the MP4EO parameters to generate a random
        instance.

    _constraints : Dict
        The information regarding the constraints to be considered.

    _preprocessed_constraint_idxs : Dict
        Decision variable indices obtained from a pre-processing
        of the considered constraints.

    theta : np.ndarray[int]
        The set of numbers of possible opportunities :math:`\vartheta_{ij}`
        in which satellite :math:`i` can acquire a desidered target
        :math:`j`.

    _n_dto : int
        The total number of Data Take Opportunities (DTO) decision
        variables.

    dto_times : np.ndarray[float]
        The set of instantaneous times :math:`t_{ij}^k` in which satellite
        :math:`i` acquire target :math:`j` at the opportunity :math:`k`.

    dto_preparation_times : np.ndarray[float]
        The set of preparation times :math:`p_{ij}^k` of satellite :math:`i`
        to gain target :math:`j` at occurrence :math:`k`.

    sigma : np.ndarray[int]
        The set of numbers of possibile opportunities :math:`\sigma_i` in
        which satellite :math:`i` can perform the downlink of an acquired
        target.

    _n_dlo : int
        The total number of DownLink Opportunities (DLO) decision
        variables.

    dlo_times : np.ndarray[float]
        The set of instantaneous times :math:`s_{ij}^r` in which satellite
        :math:`i` performes the download of target :math:`j` at the opportunity
        :math:`r`.

    dlo_preparation_times : np.ndarray[float]
        The set of preparation downlink times :math:`d_{ij}^r` of satellite
        :math:`i` to download target :math:`j` at occurence :math:`r`.

    required_target_memories : np.ndarray[float]
        The set of needed memories :math:`q_{ij}` to gain target :math:`j`
        from satellite :math:`i`.

    total_available_memories : np.ndarray[float]
        The set of total memories :math:`q_i` available from satellite
        :math:`i`.

    alpha : float
        The weight :math:`\alpha` associated to the DTO part
        of the cost function.

    gamma : float
        The weight :math:`\gamma` associated to the DTO part
        of the cost function.

    beta : float
        The weight :math:`\beta` associated to the DLO part
        of the cost function.

    omega_dto : np.ndarray[float]
        The set of DTO-binary weights in the cost function
        :math:`\omega^{\mathrm{DTO}}_{ijk}`.

    omega_dlo : np.ndarray[float]
        The set of DLO-binary weights in the cost function
        :math:`\omega^{\mathrm{DLO}}_{ijr}`.

    metrics : Dict[str, Any]
        Dictionary that contains different metrics to assess
        the hardness of this problem instance object after
        its optimization.

    _n_mp4eo_vars : int
        The total number of MP4EO decision varibles, i.e., DTO
        and DLO binaries.

    _n_vars : int
        The total number of decision variables to optimize, auxiliary
        binaries included.

    log_tag : str
        A tag to distinguish results and log files belonging
        to different simulations for the same instance.

    _log_tag_history : List[str]
        A list to store log-file tag history.

    log_dir : str | pathlib.Path
        The base directory for saving log files associated with the
        MP4EO instance.

    Details
    -------
    A generic MP4EO problem instance is defined by the minimization
    of the following linear cost function

    .. math::

        \mathcal{F}(\boldsymbol{x}, \boldsymbol{y})
        =
        \sum_{i=1}^N \sum_{j=1}^M \sum_{k=1}^{\vartheta_{ij}}
        \left(\alpha t_{ij}^k - \dfrac{\gamma}{M}\right) x_{ij}^k
        +
        \sum_{i=1}^N \sum_{j=1}^M \sum_{r=1}^{\sigma_i}
        \beta s_{ij}^r y_{ij}^r
        =
        \sum_{i=1}^N \sum_{j=1}^M \sum_{k=1}^{\vartheta_{ij}}
        \omega^{\mathrm{DTO}}_{ijk} x_{ijk}
        +
        \sum_{i=1}^N \sum_{j=1}^M \sum_{r=1}^{\sigma_i}
        \omega^{mathrm{DLO}}_{ijr} y_{ijr}

    where :math:`x_{ijk} \in \{0, 1\}` and :math:`y_{jir} \in \{0, 1\}`
    are DTO and DLO decision variables respectively.
    In the ``test``-type instances, we set all
    :math:`\omega^{\mathrm{DTO}}_{ijk} x_{ijk} = 1` and all
    :math:`\omega^{\mathrm{DLO}}_{ijr} y_{ijr} = 1` without loss
    of generality.

    In this project, we consider the following (at most quadratic)
    equality and inequality constraints:

    - **(1) One aquisition attempt per target**

      .. math::

        \sum_{i=1}^N \sum_{k=1}^{\vartheta_{ij}} x_{ij}^k - 1 = 0
        \qquad \forall j \in [1, M]

    - **(2) One download attempt per target**

      .. math::

        \sum_{i=1}^N\sum_{r=1}^{\sigma_{i}} y_{ij}^r - 1 = 0
        \qquad \forall j \in [1, m]

    - **(3) Each acquisition has to be downloaded**

      .. math::

        \begin{split}
        &\sum_{k=1}^{\vartheta_{ij}} x_{ij}^k
        - \sum_{r=1}^{\sigma_i} y_{ij}^r = 0 \\
        &\forall i \in [1, N], \ \forall j \in [1, M]

    - **(4) Download preceeds acquisition**

      .. math::

        \begin{split}
        &\sum_{k=1}^{\vartheta_{ij}} x_{ij}^k t_{ij}^k
        - \sum_{r=1}^{\sigma_i} y_{ij}^r s_{ij}^r \leq 0
        \\
        &\forall i \in [1, N], \ \forall j \in [1, M]

    - **(5) Acquisition attempts only possible if preceding acquisitions allow for sufficient preparation time**

      .. math::

        \begin{split}
        &x_{is}^k p_{is}^k -
        \left\lvert{x_{is}^k t_{is}^k - x_{ij}^{k^{\prime}}
        t_{ij}^{k^{\prime}}}\right\rvert \leq 0
        \\
        &\forall i \in [1, N], \ \forall j,s \in [1, M],
        \\
        &\forall k \in [1, \vartheta_{is}],
        \ \forall k^{\prime} \in [1, \vartheta_{ij}]
        \end{split}

    - **(6) Download attempts only possible if preceding download allow for sufficient preparation time**

      .. math::

        \begin{split}
        &y_{ig}^r d_{ig}^r -
        \left\lvert{y_{ig}^r s_{ig}^r - y_{ij}^{r^{\prime}}
        s_{ij}^{r^{\prime}}}\right\rvert \leq 0
        \\
        &\forall i \in [1, N], \ \forall j,g \in [1, M],
        \\
        &\forall r \in [1, \sigma_{i}], \forall r^{\prime} \in [1, \sigma_{i}]
        \end{split}

    - **(7) Memory is bounded from below and above**

      .. math::

        \begin{split}
        &\sum_{j=1}^M \left( \sum_{k=1}^{k^{\prime}} x_{ij}^k
        - \sum_{r=1}^{r^{\prime}} y_{ij}^r \right)q_{ij} - q_i^{tot} \leq 0
        \\
        &\forall i \in [1, N],
        \\
        &\forall k^{\prime} \in [1, \vartheta_{ij}]
        \ \forall r^{\prime} \in [1, \sigma_{i}] :
        t_{ij}^{k^{\prime }} \leq s_{ij}^{r^{\prime}}
        \end{split}

    Any MP4EO problem instance is defined by all the parameters appearing
    in the above cost function and constraints.

    Notes
    -----
    Additional constraints can be added, removed, and modified after
    the instance is created using the appropriate methods of this class.

    The parameters for generating random MP4EO instances follow
    a uniform probability distribution. If testing requires a
    different underlying probability distribution in the future,
    a new class parameter can be introduced to support this functionality.

    For random testing instances created by this constructor, all
    parameters are selected as integers, including those related to
    acquisition times and memory. If testing instances with real-valued
    parameters is required—such as in realistic use-case scenarios, this
    additional option can be easily introduced.
    """

    # ---------------------------------------------------------------------- #
    #                              Constructor                               #
    # ---------------------------------------------------------------------- #
    def __init__(
        self,
        n_satellites,
        n_targets,
        theta,
        n_dto,
        dto_times,
        dto_preparation_times,
        sigma,
        n_dlo,
        dlo_times,
        dlo_preparation_times,
        required_target_memories,
        total_available_memories,
        alpha,
        beta,
        gamma,
        omega_dto,
        omega_dlo,
        iparam_range,
        binaries_arrangement="target-wise",
        constraints=None,
    ):
        """
        Initializes an MP4EO problem instance by providing all the
        parameters.
        """

        # Attributes from parameters
        self._N = n_satellites
        self._M = n_targets
        self.log_tag = f"{self._N:04d}_{self._M:04d}"

        self.theta = theta
        self._n_dto = n_dto
        self.dto_times = dto_times
        self.dto_preparation_times = dto_preparation_times

        self.sigma = sigma
        self._n_dlo = n_dlo
        self.dlo_times = dlo_times
        self.dlo_preparation_times = dlo_preparation_times

        self.required_target_memories = required_target_memories
        self.total_available_memories = total_available_memories

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega_dto = omega_dto
        self.omega_dlo = omega_dlo
        self.iparam_ranges = iparam_range

        self._n_slack_vars = 0
        self._binaries_per_slack = None
        self._binaries_arrangement = binaries_arrangement
        self._build_binary_sg_index_mapping()
        self._slack_bin_sg_index_map = {}

        if constraints is None:
            self._constraints = {f"C{pp}": (False, None) for pp in range(1, 8)}
        else:
            self._constraints = constraints
        self._preprocessed_constraint_idxs = {
            f"C{pp}": (False, None) for pp in range(4, 8)
        }
        self.preprocess_constraints()

        # Check instance parameters
        if self.iparam_ranges is not None:
            sat_range = self.iparam_ranges["N_SAT"]
            tar_range = self.iparam_ranges["N_TAR"]
            if not sat_range[0] <= n_satellites <= sat_range[1]:
                raise MP4EOInstanceConstructionError(
                    "Number of satellites in the constellation"
                    f" not in [{sat_range[0]}, {sat_range[1]}]. Reset it!"
                )
            if not tar_range[0] <= n_targets <= tar_range[1]:
                raise MP4EOInstanceConstructionError(
                    "Number of targets to be acquired "
                    f" not in [{tar_range[0]}, {tar_range[1]}]. Reset it!"
                )

        # Other attributes
        self._n_mp4eo_vars = n_dto + n_dlo
        self._n_vars = self._n_mp4eo_vars + self._n_slack_vars

        # Pre-compute metrics based on instance object parameters
        self._compute_metrics()

        self.log_tag += f"_{n_dto:05d}_{n_dlo:05d}"
        for pp, (_, (cflag, _)) in enumerate(self._constraints.items()):
            self.log_tag += "_"
            if cflag:
                self.log_tag += f"C{pp + 1}"

        self._log_tag_history = [self.log_tag]
        self.log_dir = Path(f"MP4EO_RESULTS/instance/")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    #                               Properties                               #
    # ---------------------------------------------------------------------- #
    @property
    def n_satellites(self):
        """Read-only access to :math:`N`."""
        return self._N

    @property
    def n_targets(self):
        """Read-only access to :math:`M`."""
        return self._M

    @property
    def n_data_take_opportunity(self):
        """Read-only access to :attr:`~MP4EOInstance._n_dto`."""
        return self._n_dto

    @property
    def n_downlink_opportunity(self):
        """Read-only access to :attr:`~MP4EOInstance._n_dlo`."""
        return self._n_dlo

    @property
    def n_slack_binaries(self):
        """Read-only access to :attr:`~MP4EOInstance._n_slack_vars`."""
        return self._n_slack_vars

    @property
    @ensure_init("_binaries_per_slack")
    def binaries_per_slack(self):
        """Read-only access to :attr:`~MP4EOInstance._binaries_per_slack`."""
        return self._binaries_per_slack

    @property
    def n_mp4eo_decision_variables(self):
        """Read-only access to the number of only MP4EO binaries."""
        return self._n_mp4eo_vars

    @property
    def n_decision_variables(self):
        """Read-only access to the number of decision variables."""
        return self._n_vars

    @n_decision_variables.setter
    def n_decision_variables(self, updated_value):
        """Setter for the number of decision variables."""
        if not isinstance(updated_value, int) or updated_value <= 0:
            raise ValueError(
                "The number of decision variables must be a positve integer."
            )
        self._n_vars = updated_value

    @property
    def binaries_arrangement(self):
        """Read-only access to :attr:`~MP4EOInstance._binaries_arrangement."""
        return self._binaries_arrangement

    @property
    def dto_bin_sg_index_map(self):
        """Read-only access to :attr:`~MP4EOInstance._dto_bin_sg_index_map."""
        return self._dto_bin_sg_index_map

    @property
    def dlo_bin_sg_index_map(self):
        """
        Read-only access to :attr:`~MP4EOInstance._dlo_bin_sg_index_map.
        """
        return self._dlo_bin_sg_index_map

    @property
    def slack_bin_sg_index_map(self):
        """
        Read-only access to :attr:`~MP4EOInstance._slack_bin_sg_index_map.
        """
        return self._slack_bin_sg_index_map

    @property
    def constraints(self):
        """Read-only access to problem constraints."""
        return self._constraints

    @property
    def preprocessed_constraints(self):
        """
        Read-only access to the dictionary that contains the
        set of indices to efficiently force constraints.
        """
        return self._preprocessed_constraint_idxs

    @property
    def instance_type(self):
        """Distinguish explicit input arrays from caller-specified synthetic ranges."""
        return "explicit-input" if self.iparam_ranges is None else "synthetic"

    # ---------------------------------------------------------------------- #
    #                            Instance Methods                            #
    #                            **    Utils   **                            #
    # ---------------------------------------------------------------------- #
    def feval(self, mission_bitstring):
        r"""
        Evaluates the MP4EO cost function on an input mission
        bitstring ``mission_bitstring``.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a specific planned mission, i.e.,
            the set of DTO :math:`x_{ijk}` and DLO :math:`y_{ikr}`
            decision variables.

        Returns
        -------
        cost : float
            The cost :math:`\mathcal{F}` of the planned mission.
        """

        # Compute DTO contribution to the cost function
        dto_idxs = [
            self._dto_idxs_to_sg_idx(ii, jj, kk)
            for ii in range(self._N)
            for jj in range(self._M)
            for kk in range(self.thetaij(ii, jj))
        ]
        dto_values = np.array(
            [
                self.wijk(ii, jj, kk)
                for ii in range(self._N)
                for jj in range(self._M)
                for kk in range(self.thetaij(ii, jj))
            ]
        )
        dto_contrib = np.dot(dto_values, mission_bitstring[dto_idxs])

        # Compute DLO contribution to the cost function
        dlo_idxs = [
            self._dlo_idxs_to_sg_idx(ii, jj, rr)
            for ii in range(self._N)
            for jj in range(self._M)
            for rr in range(self.sigma[ii])
        ]
        dlo_values = np.array(
            [
                self.wijr(ii, jj, rr)
                for ii in range(self._N)
                for jj in range(self._M)
                for rr in range(self.sigma[ii])
            ]
        )
        dlo_contrib = np.dot(dlo_values, mission_bitstring[dlo_idxs])

        # Output
        return dto_contrib + dlo_contrib

    def check_constraints(self, mission_bitstring):
        """
        Checks all considered optimization constraints on
        an input mission bitstring ``mission_bitstring``.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a specific planned mission, i.e.,
            the set of DTO :math:`x_{ijk}` and DLO :math:`y_{ikr}`
            decision variables.

        Returns
        -------
        checks_record : Dict[str, Tuple[int]]
            The outcome of the checks as a dictionary:
            keys are constraint names, values are a
            Tuple[int, int] where the first integer
            is the number of sarisfied equations/inequalities
            for the corresponding constraint, while the
            second integer represents the total number of
            equations/inequalities to be checked for that
            constraint.
        """

        checks_record = {}
        cmethods = {
            # One acquisition attempt per target
            "C1": self.check_constraint_1,
            # One download attempt per target
            "C2": self.check_constraint_2,
            # Each acquisition must be downloaded
            "C3": self.check_constraint_3,
            # Target acquisition precede download
            "C4": self.check_constraint_4,
            # Sufficient preparation time between two acquisitions
            "C5": self.check_constraint_5,
            # Sufficient preparation time between two downlinks
            "C6": self.check_constraint_6,
            # Available memory on board
            "C7": self.check_constraint_7,
        }

        # Check constraints when considered
        for cname, cmethod in cmethods.items():
            if self.get_constraint(cname)[0]:
                checks_record[cname] = cmethod(mission_bitstring)

        # Output
        return checks_record

    def get_mission_quality(self, mission_bitstring):
        """
        Returns the quality of a planned mission described by
        ``mission_bitstring`` in terms of total satisfied MP4EO
        constraints.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a specific planned mission, i.e.,
            the set of DTO :math:`x_{ijk}` and DLO :math:`y_{ikr}`
            decision variables.

        Returns
        -------
        quality : float
            The percentage of satisfied constraints over the
            total number of constraints.
            The output is a real number between 0 and 1.
        """

        # Check constraints and record violations
        check_dict = self.check_constraints(mission_bitstring)
        tot_satisfied, tot_constraints = tuple(
            map(sum, zip(*check_dict.values()))
        )

        # Compute the quality
        try:
            quality = round(tot_satisfied / tot_constraints, 4)
        except ZeroDivisionError:  # free optimization
            quality = 1

        # Output
        return quality

    def get_constraint(self, cname):
        """
        Read-only access to a specific problem constraint.

        Arguments
        ---------
        cname : str
            The string for recognizing the desidered
            constraint.
        """
        try:
            return self._constraints[cname]
        except KeyError:
            raise KeyError(f"MP4EO constraint {cname} does not exist!")

    def get_preprocessed_constraint_flag(self, cname):
        """
        Read-only access to constraint flag for
        preprocessed set of indices.

        Arguments
        ---------
        cname : str
            The string for recognizing the desidered
            constraint.
        """
        try:
            return self._preprocessed_constraint_idxs[cname][0]
        except KeyError:
            raise KeyError(
                f"MP4EO preprocessed constraint {cname} does not exist!"
            )

    def get_preprocessed_constraint_idxs(self, cname):
        """
        Read-only access to preprocessed set of indices for a specific
        constraint. Note that this is only an access method for the
        corresponding class instance attribute, and it cannot be used
        to, e.g., enforce constraints as penalty within the QUBO reformu-
        lation.

        Arguments
        ---------
        cname : str
            The string for recognizing the desidered
            constraint.
        """
        try:
            return self._preprocessed_constraint_idxs[cname][1]
        except KeyError:
            raise KeyError(
                f"MP4EO preprocessed constraint {cname} does not exist!"
            )

    def add_constraint(self, cname, clambda):
        """
        Adds a specific constraint to be considered
        during the optimization.

        Arguments
        ---------
        cname : str
            The name of the constraint to be added.

        clambda : float
            The penalty constant weight for the specific
            constraint to be added.

        Returns
        -------
        :obj:`None`
        """

        if cname in self._constraints:
            # Add the new constraint
            self._constraints[cname] = (True, clambda)

            # Pre-processed the constraint if needed
            if cname in self._preprocessed_constraint_idxs:
                prep_match = re.match(r"C(\d+)", cname)
                if prep_match:
                    cnum = prep_match.group(1)
                    prep_method_name = f"preprocess_constraint_{cnum}"
                    prep_method = getattr(self, prep_method_name, None)
                    if callable(prep_method):
                        prep_method()

            # Update the tag for the log files
            cmatch = re.search(
                r"(C\d+)+", self.log_tag
            )  # Search for the constraint sub-tag
            if cmatch:
                constraint_subtag = cmatch.group(
                    0
                )  # Extract current constraints
                existing_constraints = re.findall(r"C\d+", constraint_subtag)

                if (
                    cname not in existing_constraints
                ):  # Insert the new constraint if not already present
                    existing_constraints.append(cname)
                    existing_constraints.sort(key=lambda x: int(x[1:]))

                updated_subtag = "".join(existing_constraints)
                self.log_tag = self.log_tag.replace(
                    constraint_subtag, updated_subtag
                )
            else:
                self.log_tag += cname

            # Update tag history
            self._log_tag_history.append(self.log_tag)
        else:
            raise KeyError(f"MP4EO constraint {cname} does not exist!")

    def remove_constraint(self, cname):
        """
        Removes a specific constraint to be ignored
        during the optimization.

        Arguments
        ---------
        cname : str
            The name of the constraint to be added.

        Returns
        -------
        :obj:`None`
        """

        if cname in self._constraints:
            # Remove the chosen constraint
            self._constraints[cname] = (False, None)

            # Update also the pre-processed constraint dictionary
            if cname in self._preprocessed_constraint_idxs:
                self._preprocessed_constraint_idxs[cname] = (False, None)

            # Update the tag for the log files
            cmatch = re.search(
                r"(C\d+)+", self.log_tag
            )  # Search for the constraint sub-tag
            if cmatch:
                constraint_subtag = cmatch.group(
                    0
                )  # Extract current constraints
                existing_constraints = re.findall(r"C\d+", constraint_subtag)

                if cname in existing_constraints:  # Remove the constraint
                    existing_constraints.remove(cname)

                updated_subtag = "".join(existing_constraints)
                self.log_tag = self.log_tag.replace(
                    constraint_subtag, updated_subtag
                )
        else:
            raise KeyError(f"MP4EO constraint {cname} does not exist!")

    def update_constraint(self, cname, clambda):
        """
        Updates the QUBO penalty constant of a specific constraint.

        Arguments
        ---------
        cname : str
            The name of the constraint to be updated.

        clambda : float
            The penalty constant weight for the specific
            constraint to be updated.

        Returns
        -------
        :obj:`None`
        """

        if cname in self._constraints:
            old_cflag = self._constraints[cname][0]
            self._constraints[cname] = (old_cflag, clambda)
        else:
            raise KeyError(f"MP4EO constraint {cname} does not exist!")

    def check_constraint_1(self, mission_bitstring):
        r"""
        Check constraint (1) on an input mission bitstring
        ``mission_bitstring``.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a specific planned mission, i.e.,
            the set of DTO :math:`x_{ijk}` and DLO :math:`y_{ikr}`
            decision variables.

        Returns
        -------
        check_record : Tuple[int, int]
            The outcome of the check: the first integer
            is the number of sarisfied equations for
            constraint (1), the second integer represents
            the total number of equations to be checked.
        """

        # Check constraint (1)
        satisfied = 0
        for jj in range(self._M):
            bin_sum = sum(
                mission_bitstring[self._dto_idxs_to_sg_idx(ii, jj, kk)]
                for ii in range(self._N)
                for kk in range(self.thetaij(ii, jj))
            )
            if bin_sum == 1:
                satisfied += 1

        # Output
        return satisfied, self._M

    def check_constraint_2(self, mission_bitstring):
        r"""
        Check constraint (2) on an input mission bitstring
        ``mission_bitstring``.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a specific planned mission, i.e.,
            the set of DTO :math:`x_{ijk}` and DLO :math:`y_{ikr}`
            decision variables.

        Returns
        -------
        check_record : Tuple[int, int]
            The outcome of the check: the first integer
            is the number of sarisfied equations for
            constraint (2), the second integer represents
            the total number of equations to be checked.
        """

        # Check constraint (2)
        satisfied = 0
        for jj in range(self._M):
            bin_sum = sum(
                mission_bitstring[self._dlo_idxs_to_sg_idx(ii, jj, rr)]
                for ii in range(self._N)
                for rr in range(self.sigma[ii])
            )
            if bin_sum == 1:
                satisfied += 1

        # Output
        return satisfied, self._M

    def check_constraint_3(self, mission_bitstring):
        r"""
        Check constraint (3) on an input mission bitstring
        ``mission_bitstring``.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a specific planned mission, i.e.,
            the set of DTO :math:`x_{ijk}` and DLO :math:`y_{ikr}`
            decision variables.

        Returns
        -------
        check_record : Tuple[int, int]
            The outcome of the check: the first integer
            is the number of sarisfied equations for
            constraint (3), the second integer represents
            the total number of equations to be checked.
        """

        # Check constraint (3)
        satisfied = 0
        for ii in range(self._N):
            for jj in range(self._M):
                dto_bin_sum = sum(
                    mission_bitstring[self._dto_idxs_to_sg_idx(ii, jj, kk)]
                    for kk in range(self.thetaij(ii, jj))
                )
                dlo_bin_sum = sum(
                    mission_bitstring[self._dlo_idxs_to_sg_idx(ii, jj, rr)]
                    for rr in range(self.sigma[ii])
                )
                if dto_bin_sum == dlo_bin_sum:
                    satisfied += 1

        # Output
        return satisfied, self._N * self._M

    def check_constraint_4(self, mission_bitstring):
        r"""
        Check constraint (4) on an input mission bitstring
        ``mission_bitstring``.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a specific planned mission, i.e.,
            the set of DTO :math:`x_{ijk}` and DLO :math:`y_{ikr}`
            decision variables.

        Returns
        -------
        check_record : Tuple[int, int]
            The outcome of the check: the first integer
            is the number of sarisfied inequalities for
            constraint (4), the second integer represents
            the total number of inequalities to be checked.
        """

        # Check constraint (4)
        satisfied = 0
        for ii in range(self._N):
            for jj in range(self._M):
                dto_bin_sum = sum(
                    self.tijk(ii, jj, kk)
                    * mission_bitstring[self._dto_idxs_to_sg_idx(ii, jj, kk)]
                    for kk in range(self.thetaij(ii, jj))
                )
                dlo_bin_sum = sum(
                    self.sijr(ii, jj, rr)
                    * mission_bitstring[self._dlo_idxs_to_sg_idx(ii, jj, rr)]
                    for rr in range(self.sigma[ii])
                )
                if dto_bin_sum <= dlo_bin_sum:
                    satisfied += 1

        # Output
        return satisfied, self._N * self._M

    def check_constraint_5(self, mission_bitstring):
        r"""
        Check constraint (5) on an input mission bitstring
        ``mission_bitstring``.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a specific planned mission, i.e.,
            the set of DTO :math:`x_{ijk}` and DLO :math:`y_{ikr}`
            decision variables.

        Returns
        -------
        check_record : Tuple[int, int]
            The outcome of the check: the first integer
            is the number of sarisfied inequalities for
            constraint (5), the second integer represents
            the total number of inequalities to be checked.
        """

        # Check constraint (5)
        satisfied = 0
        ntot = 0
        for ii in range(self._N):
            for ss in range(self._M):
                for jj in range(self._M):
                    for kk in range(self.thetaij(ii, ss)):
                        bin_idx_1 = self._dto_idxs_to_sg_idx(ii, ss, kk)

                        for kkp in range(self.thetaij(ii, jj)):
                            bin_idx_2 = self._dto_idxs_to_sg_idx(ii, jj, kkp)

                            if bin_idx_1 != bin_idx_2:
                                ntot += 1

                                value_to_check = (
                                    self.pijk(ii, ss, kk)
                                    * mission_bitstring[bin_idx_1]
                                )
                                value_to_check -= abs(
                                    self.tijk(ii, ss, kk)
                                    * mission_bitstring[bin_idx_1]
                                    - self.tijk(ii, jj, kkp)
                                    * mission_bitstring[bin_idx_2]
                                )

                                if value_to_check <= 0:
                                    satisfied += 1

        # Output
        return satisfied, ntot

    def check_constraint_6(self, mission_bitstring):
        r"""
        Check constraint (6) on an input mission bitstring
        ``mission_bitstring``.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a specific planned mission, i.e.,
            the set of DTO :math:`x_{ijk}` and DLO :math:`y_{ikr}`
            decision variables.

        Returns
        -------
        check_record : Tuple[int, int]
            The outcome of the check: the first integer
            is the number of sarisfied inequalities for
            constraint (6), the second integer represents
            the total number of inequalities to be checked.
        """

        # Check constraint (6)
        satisfied = 0
        ntot = 0
        for ii in range(self._N):
            for gg in range(self._M):
                for jj in range(self._M):
                    for rr in range(self.sigma[ii]):
                        bin_idx_1 = self._dlo_idxs_to_sg_idx(ii, gg, rr)

                        for rrp in range(self.sigma[ii]):
                            bin_idx_2 = self._dlo_idxs_to_sg_idx(ii, jj, rrp)

                            if bin_idx_1 != bin_idx_2:
                                ntot += 1

                                value_to_check = (
                                    self.dijr(ii, gg, rr)
                                    * mission_bitstring[bin_idx_1]
                                )
                                value_to_check -= abs(
                                    self.sijr(ii, gg, rr)
                                    * mission_bitstring[bin_idx_1]
                                    - self.sijr(ii, jj, rrp)
                                    * mission_bitstring[bin_idx_2]
                                )

                                if value_to_check <= 0:
                                    satisfied += 1

        # Output
        return satisfied, ntot

    def check_constraint_7(self, mission_bitstring=None):
        r"""
        Check constraint (7) on an input mission bitstring
        ``mission_bitstring``.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a specific planned mission, i.e.,
            the set of DTO :math:`x_{ijk}` and DLO :math:`y_{ikr}`
            decision variables.

        Returns
        -------
        check_record : Tuple[int, int]
            The outcome of the check: the first integer
            is the number of sarisfied inequalities for
            constraint (7), the second integer represents
            the total number of inequalities to be checked.
        """

        # Check constraint (6)
        satisfied = 0
        for ii in range(self._N):
            total_memory_occupied = 0

            for jj in range(self._M):
                target_memory_occupied = sum(
                    self.qij(ii, jj)
                    * (
                        mission_bitstring[self._dto_idxs_to_sg_idx(ii, jj, kk)]
                        - mission_bitstring[
                            self._dlo_idxs_to_sg_idx(ii, jj, rr)
                        ]
                    )
                    for kk in range(self.thetaij(ii, jj))
                    for rr in range(self.sigma[ii])
                    if self.tijk(ii, jj, kk) <= self.sijr(ii, jj, rr)
                )
                total_memory_occupied += target_memory_occupied

            if total_memory_occupied <= self.total_available_memories[ii]:
                satisfied += 1

        # Output
        return satisfied, self._N

    def update_tag(self, new_info):
        """
        Updates with new information ``new_info`` the
        instance attribute :attr:`~MP4EOInstance.log_tag`.

        Arguments
        ---------
        new_info : str
            The new information to append at the end
            of :attr:`~MP4EOInstance.log_tag`.

        Returns
        -------
        :obj:`None`
        """

        # Ensure string data type
        if not isinstance(new_info, str):
            new_info = str(new_info)

        # Add underscore if not already present
        if not new_info.startswith("_"):
            new_info = f"_{new_info}"

        # Save the current tag in history
        self._log_tag_history.append(self.log_tag)

        # Update the tag
        self.log_tag += new_info

    def revert_to_previous_tag(self):
        """
        Reverts the :attr:`~MP4EOInstance.log_tag` attribute to its
        previous value stored in the history stack.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Raises
        ------
        :exc:`ValueError`
            If there is no previous tag to revert to.
        """

        if len(self._log_tag_history) <= 1:
            raise ValueError("No previous log-file tag to revert to.")

        # Pop the last tag from the history and revert
        self.log_tag = self._log_tag_history.pop()

    def reset_tag(self):
        """
        Resets the :attr:`~MP4EOInstance.log_tag` attribute to its
        original value (first value in history).

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`
        """

        if self._log_tag_history:
            self.log_tag = self._log_tag_history[0]
            # Clear history except the original tag
            self._log_tag_history = [self.log_tag]

    def print_details(self):
        """
        Prints details of the MP4EO problem instance ono standard output.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`
        """

        # Heading
        head = "MP4EO problem instance details"
        sep_line = "-" * len(head)
        print(f"{sep_line}\n{head}\n{sep_line}\n")

        # Parameters of the mission
        mission_details = {
            "Number of satellites": self.n_satellites,
            "Number of targets": self.n_targets,
            "Number of Data Take Opportunity (DTO) binaries": self.n_data_take_opportunity,
            "Number of Downlink Opportunity (DLO) binaries": self.n_downlink_opportunity,
            "Total number of MP4EO decision variables": self.n_mp4eo_decision_variables,
        }
        for description, value in mission_details.items():
            print(f"{description} <-- {value}")

        # Constraints of the mission
        print("\nConstraints to be considered during the optimization:")
        if not any(cinfo[0] for cinfo in self.constraints.values()):
            print("\tFree optimization")
        else:
            constraint_descriptions = {
                "C1": "One acquisition attempt per target",
                "C2": "One download attempt per target",
                "C3": "Each acquisition must be downloaded",
                "C4": "Target acquisition precedes download",
                "C5": "Sufficient preparation time between two acquisitions",
                "C6": "Sufficient preparation time between two downlinks",
                "C7": "Available memory on board",
            }
            for ii, (cname, descr) in enumerate(
                constraint_descriptions.items()
            ):
                if self.get_constraint(cname)[0]:
                    print(f"\t({ii+1}) {descr}")

        # Type of instance
        type_msg = {
            "synthetic": "Caller-specified synthetic ranges",
            "explicit-input": "Explicit input arrays",
        }
        print(
            f"\nProblem instance type <-- {type_msg[self.instance_type]}\n\n\n"
        )

    def _compute_metrics(self):
        """
        Computes and stores metrics to assess instance hardness
        a-posteriori for a given solver of this problem instance.
        The corresponding instance attribute :attr:`~MP4EOInstance.metrics`
        could be further update later from specific solvers and algorithms.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`
        """

        self.metrics = {}

        # Spread of cost-function weights
        # Measures disparity in cost coefficients
        all_weights = np.concatenate((self.omega_dto, self.omega_dlo))
        mean_weight = np.mean(all_weights)
        weight_diffs = np.abs(all_weights[:, None] - all_weights[None, :])
        weight_spread = np.sum(weight_diffs) / (
            2 * len(all_weights) * mean_weight
        )

        # Time-overlap density
        # Two acquisitions/downlink times are considered overlapping if
        # they occur within a small time window, dependig on the system
        # reoslution.
        dto_count = dlo_count = 0
        dto_deltat = dlo_deltat = 1

        dto_times = sorted(self.dto_times)
        dlo_times = sorted(self.dlo_times)

        for ii in range(self._n_dto):
            jj = ii + 1
            while (
                jj < self._n_dto
                and dto_times[jj] - dto_times[ii] <= dto_deltat
            ):
                dto_count += 1
                jj += 1

        for ii in range(self._n_dlo):
            jj = ii + 1
            while (
                jj < self._n_dlo
                and dlo_times[jj] - dlo_times[ii] <= dlo_deltat
            ):
                dlo_count += 1
                jj += 1

        dto_time_overlap = 2 * dto_count / (self._n_dto * (self._n_dto - 1))
        dlo_time_overlap = 2 * dlo_count / (self._n_dlo * (self._n_dlo - 1))

        # Store computed metrics
        self.metrics["weight_spread"] = weight_spread
        self.metrics["dto_time_overlap_density"] = dto_time_overlap
        self.metrics["dlo_time_overlap_density"] = dlo_time_overlap

    # ---------------------------------------------------------------------- #
    #                             Static Methods                             #
    # ---------------------------------------------------------------------- #
    # TODO: remove this static method
    # TODO: Temporary checks on duplicates in preprocessed set of indices
    # TODO: for constraints
    @staticmethod
    def has_duplicates(lst):
        count = Counter(lst)
        return any(value > 1 for value in count.values())

    @staticmethod
    def _get_mp4eo_parameters_ranges(instance_type, parameter_ranges=None):
        """Use caller-specified synthetic ranges; no industrial defaults exist."""
        if instance_type not in ("synthetic", "test"):
            raise MP4EOInstanceConstructionError(
                "Only independently specified synthetic ranges are supported."
            )
        return get_synthetic_ranges(parameter_ranges)

    @staticmethod
    def _read_dto_params_from_tasi_csv(*args, **kwargs):
        """Industrial data adapter excluded from the public synthetic release."""
        raise NotImplementedError(
            "Industrial CSV adapters are excluded; construct MP4EOInstance "
            "from independently specified synthetic arrays instead."
        )

    @staticmethod
    def _read_dlo_params_from_tasi_csv(*args, **kwargs):
        """Industrial data adapter excluded from the public synthetic release."""
        raise NotImplementedError(
            "Industrial CSV adapters are excluded; construct MP4EOInstance "
            "from independently specified synthetic arrays instead."
        )

    # ---------------------------------------------------------------------- #
    #                            Instance Methods                            #
    #                **    Quadratic Program Formulation   **                #
    # ---------------------------------------------------------------------- #
    @cache
    def thetaij(self, sat_idx, tar_idx):
        r"""
        Getter method for the number of DTOs :math:`\vartheta_{ij}`.

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the satellite
            in :math:`\vartheta_{ij}`.

        tar_idx : int
            The index :math:`j` that identifies the target in
            :math:`\vartheta_{ij}`.

        Returns
        -------
        :math:`\vartheta_{ij}` : int
            The number of data-take opportunity for a given
            satellite :math:`i` and a given target :math:`j`.
        """

        assert (
            0 <= sat_idx < self._N
        ), f"Invalid index {sat_idx} for {self._N} satellites."
        assert (
            0 <= tar_idx < self._M
        ), f"Invalid index {tar_idx} for {self._M} targets."
        return self.theta[sat_idx * self._M + tar_idx]

    @cache
    def tijk(self, sat_idx, tar_idx, dto_idx):
        r"""
        Getter method for DTO acquisition times :math:`t_{ijk}`.

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the
            satellite in the acquisition times :math:`t_{ijk}`.

        tar_idx : int
            The index :math:`j` that identifies the
            target in the acquisition times :math:`t_{ijk}`.

        dto_idx : int
            The index :math:`k` that identifies the
            data-take opportunity in the acquisition
            times :math:`t_{ijk}`.

        Returns
        -------
        :math:`t_{ijk}` : float
            The data-take opportunity acquisition time
            identified by the index triplet :math:`\{i, j, k\}`.
        """

        assert (
            0 <= dto_idx < self.thetaij(sat_idx, tar_idx)
        ), f"Invalid index {dto_idx} for the selected DTO."
        t_idx = sum(
            self.thetaij(iip, jj)
            for iip in range(sat_idx)
            for jj in range(self._M)
        )
        t_idx += sum(self.thetaij(sat_idx, jjp) for jjp in range(tar_idx))
        return self.dto_times[int(t_idx) + dto_idx]

    @cache
    def pijk(self, sat_idx, tar_idx, dto_idx):
        r"""
        Getter method for DTO preparation times :math:`p_{ijk}`.

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the
            satellite in the preparation times :math:`p_{ijk}`.

        tar_idx : int
            The index :math:`j` that identifies the
            target in the preparation times :math:`p_{ijk}`.

        dto_idx : int
            The index :math:`k` that identifies the
            data-take opportunity in the preparation
            times :math:`p_{ijk}`.

        Returns
        -------
        :math:`p_{ijk}` : float
            The data-take opportunity preparation time
            identified by the index triplet :math:`\{i, j, k\}`.
        """

        assert (
            0 <= dto_idx < self.thetaij(sat_idx, tar_idx)
        ), f"Invalid index {dto_idx} for the selected DTO."
        p_idx = sum(
            self.thetaij(iip, jj)
            for iip in range(sat_idx)
            for jj in range(self._M)
        )
        p_idx += sum(self.thetaij(sat_idx, jjp) for jjp in range(tar_idx))
        return self.dto_preparation_times[int(p_idx) + dto_idx]

    @cache
    def wijk(self, sat_idx, tar_idx, dto_idx):
        r"""
        Getter method for DTO-binary weights :math:`\omega^{\mathrm{DTO}}_{ijk}`.

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies
            the satellite in the DTO-binary weight
            :math:`\omega^{\mathrm{DTO}}_{ijk}`.

        tar_idx : int
            The index :math:`j` that identifies
            the target in the DTO-binary weight
            :math:`\omega^{\mathrm{DTO}}_{ijk}`.

        dto_idx : int
            The index :math:`k` that identifies the
            data-take opportunity in the DTO-binary
            weight :math:`\omega^{\mathrm{DTO}}_{ijk}`.

        Returns
        -------
        :math:`\omega^{\mathrm{DTO}}_{ijk}` : float
            The DTO-binary weight :math:`\omega^{\mathrm{DTO}}_{ijk}`
            identified by the index triplet :math:`\{i, j, k\}`.
        """

        assert (
            0 <= dto_idx < self.thetaij(sat_idx, tar_idx)
        ), f"Invalid index {dto_idx} for the selected DTO."
        t_idx = sum(
            self.thetaij(iip, jj)
            for iip in range(sat_idx)
            for jj in range(self._M)
        )
        t_idx += sum(self.thetaij(sat_idx, jjp) for jjp in range(tar_idx))
        return self.omega_dto[int(t_idx) + dto_idx]

    @cache
    def sijr(self, sat_idx, tar_idx, dlo_idx):
        r"""
        Getter method for DLO download times :math:`s_{ijr}`.

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the
            satellite in the downlink times :math:`s_{ijr}`.

        tar_idx : int
            The index :math:`j` that identifies the
            target in the downlink times :math:`s_{ijr}`.

        dlo_idx : int
            The index :math:`r` that identifies the
            downlink opportunity in the download times
            :math:`s_{ijr}`.

        Returns
        -------
        :math:`s_{ijr}` : float
            The downlink opportunity download time identified
            by the index triplet :math:`\{i, j, r\}`.
        """

        assert (
            0 <= sat_idx < self._N
        ), f"Invalid index {sat_idx} for {self._N} satellites."
        assert (
            0 <= tar_idx < self._M
        ), f"Invalid index {tar_idx} for {self._M} targets."
        assert (
            0 <= dlo_idx < self.sigma[sat_idx]
        ), f"Invalid index {dlo_idx} for the selected DLO."
        s_idx = self._M * sum(self.sigma[iip] for iip in range(sat_idx))
        s_idx += tar_idx * self.sigma[sat_idx]
        return self.dlo_times[int(s_idx) + dlo_idx]

    @cache
    def dijr(self, sat_idx, tar_idx, dlo_idx):
        r"""
        Getter method for DLO preparation times :math:`d_{ijr}`.

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the
            satellite in the downlink preparation times
            :math:`d_{ijr}`.

        tar_idx : int
            The index :math:`j` that identifies the
            target in the downlink preparation times
            :math:`d_{ijr}`.

        dlo_idx : int
            The index :math:`r` that identifies the
            downlink opportunity in the download
            preparation times :math:`d_{ijr}`.

        Returns
        -------
        :math:`d_{ijr}` : float
            The downlink opportunity preparation time
            identified by the index triplet :math:`\{i, j, r\}`.
        """

        assert (
            0 <= sat_idx < self._N
        ), f"Invalid index {sat_idx} for {self._N} satellites."
        assert (
            0 <= tar_idx < self._M
        ), f"Invalid index {tar_idx} for {self._M} targets."
        assert (
            0 <= dlo_idx < self.sigma[sat_idx]
        ), f"Invalid index {dlo_idx} for the selected DLO."
        d_idx = self._M * sum(self.sigma[iip] for iip in range(sat_idx))
        d_idx += tar_idx * self.sigma[sat_idx]
        return self.dlo_preparation_times[int(d_idx) + dlo_idx]

    @cache
    def wijr(self, sat_idx, tar_idx, dlo_idx):
        r"""
        Getter method for DLO-binary weights :math:`\omega^{\mathrm{DLO}}_{ijr}`.

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies
            the satellite in the DLO-binary weight
            :math:`\omega^{\mathrm{DLO}}_{ijr}`.

        tar_idx : int
            The index :math:`j` that identifies the
            target in the DLO-binary weight
            :math:`\omega^{\mathrm{DLO}}_{ijr}`.

        dlo_idx : int
            The index :math:`r` that identifies the
            downlink opportunity in the DLO-binary
            weight :math:`\omega^{\mathrm{DLO}}_{ijr}`.

        Returns
        -------
        :math:`\omega^{\mathrm{DLO}}_{ijr}` : float
            The DLO-binary weight :math:`\omega^{\mathrm{DLO}}_{ijr}`
            identified by the index triplet :math:`\{i, j, r\}`.
        """

        assert (
            0 <= sat_idx < self._N
        ), f"Invalid index {sat_idx} for {self._N} satellites."
        assert (
            0 <= tar_idx < self._M
        ), f"Invalid index {tar_idx} for {self._M} targets."
        assert (
            0 <= dlo_idx < self.sigma[sat_idx]
        ), f"Invalid index {dlo_idx} for the selected DLO."
        d_idx = self._M * sum(self.sigma[iip] for iip in range(sat_idx))
        d_idx += tar_idx * self.sigma[sat_idx]
        return self.omega_dlo[int(d_idx) + dlo_idx]

    @cache
    def qij(self, sat_idx, tar_idx):
        r"""
        Getter method for the required memories to acquire
        the mission targets :math:`q_{ij}`.

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the satellite
            in :math:`q_{ij}`.

        tar_idx : int
            The index :math:`j` that identifies the target in
            :math:`q_{ij}`.

        Returns
        -------
        :math:`q_{ij}` : float
            The needed memory for a given satellite :math:`i`
            to acquire a given target :math:`j`.
        """

        assert (
            0 <= sat_idx < self._N
        ), f"Invalid index {sat_idx} for {self._N} satellites."
        assert (
            0 <= tar_idx < self._M
        ), f"Invalid index {tar_idx} for {self._M} targets."
        return self.required_target_memories[sat_idx * self._M + tar_idx]

    @cache
    def nslacki(self, sat_idx):
        r"""
        Getter method for the number of binaries needed
        to represent the i-th auxiliary slack variable
        :math:`s_i` (trivial qubitization).

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the satellite
            in :math:`\Omega_i`.
            See **Details** for more information about this
            method.

        Returns
        -------
        See the **Details** section below.

        Details
        -------
        Each auxiliary integer slack variable :math:`s_i` can
        be written in its binary representation

        .. math::

            s_i = \sum_{\alpha=0}^{\Omega_i - 1} 2^{\alpha} b_{i, \alpha}

        where :math:`b_{i, \alpha} \in \{0, 1\}` for all
        :math:`\alpha = 0, \cdots, \Omega_i - 1`, and :math:`\Omega_i`
        is the number of bits needed to represent the integer :math:`s_i`,
        i.e.,

        .. math::

            \Omega_i
            =
            \left\lfloor
            \log_2\left(s_i\right) + 1
            \right\rfloor

        This method returns the value of :math:`\Omega_i` for the specific
        :math:`i`-th slack variable :math:`s_i`.
        """

        assert (
            0 <= sat_idx < self._N
        ), f"Invalid index {sat_idx} for {self._N} satellites."
        try:
            return self.binaries_per_slack[sat_idx]
        except AttributeError:
            raise MP4EOInstanceConstructionError(
                "Constraint (7) was either not included in the optimization"
                " process or has not been pre-processed yet."
                " Please address one of these issues."
            )

    def preprocess_constraints(self):
        """
        Preprocesses inequality constraints and create sets
        of forbidden and allowed DTO and/or DLO indices
        to enforce constraints more efficiently.
        Then, update properly the corresponding class
        instance attibute.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Notes
        -----
        See each single-constraint preprocessing method for
        details.
        """

        prep_methods = {
            # Target acquisition precede download
            "C4": self.preprocess_constraint_4,
            # Sufficient preparation time between two acquisitions
            "C5": self.preprocess_constraint_5,
            # Sufficient preparation time between two downlinks
            "C6": self.preprocess_constraint_6,
            # Available memory on board
            "C7": self.preprocess_constraint_7,
        }

        # Preprocessing constraints when considered
        for cname, cmethod in prep_methods.items():
            if self.get_constraint(cname)[0]:
                cmethod()

    def update_n_slack_binaries(self):
        r"""
        Updates :attr:`~MP4EOInstance._n_slack_vars` and
        :attr:`~MP4EOInstance._binaries_per_slack` whenever
        constraint (7) is considered during the optimization
        of the corresponding QUBO function.
        Updates also the dictionary map from slack binaries
        index to the one-dimensional arrangement
        :attr:`~MP4EOInstance._slack_bin_sg_index_map`.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`
        """

        # Read the preprocessed set for constraint (7)
        cardinalities = [
            [
                len(self.get_preprocessed_c7_idxs(ii, jj))
                for jj in range(self._M)
            ]
            for ii in range(self._N)
        ]

        # Compute the number of binaries to represent each slack integer
        self._binaries_per_slack = np.array(
            [
                floor(
                    log2(
                        self.total_available_memories[ii]
                        + sum(
                            cardinalities[ii][jj] * self.qij(ii, jj)
                            for jj in range(self._M)
                        )
                        + 1
                    )
                )
                for ii in range(self._N)
            ]
        ).astype(int)

        # Update instance parameters
        self._n_slack_vars = np.sum(self._binaries_per_slack)
        self._n_vars = self._n_mp4eo_vars + self._n_slack_vars

        # Slack binaries arrangement always at the end of the
        # one-dimensional chain
        sg_idx = self._n_mp4eo_vars
        for ii in range(self._N):
            for aa in range(self._binaries_per_slack[ii]):
                self._slack_bin_sg_index_map[(ii, aa)] = sg_idx
                sg_idx += 1

    def preprocess_constraint_4(self):
        r"""
        Preprocesses constraint (4) and obtain a set
        of forbidden DTO and/or DLO indices to simplify
        the enforcing of the constraint when the QUBO
        mapping is used.
        This method directly changes the instance dictionary
        attribute :attr:`~MP4EOInsttance._preprocessed_constraint_idxs`.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        The set of forbidden DTO-DLO indices to enforce constraint
        (4) is defined as:

        .. math::

            \mathcal{R}_{ij}
            =
            \left\{
                (k, r) \quad \vert \quad t_{ij}^k > s_{ij}^r
            \right\}

        for all :math:`i = 1, \cdots, N` and :math:`j = 1, \cdots, M`.

        Notes
        -----
        The set of indices is flatten out over :math:`i` and :math:`j`
        as for :math:`\vartheta_{ij}`.
        """

        forbidden_idxs = [
            [
                (kk, rr)
                for kk in range(self.thetaij(ii, jj))
                for rr in range(self.sigma[ii])
                if self.tijk(ii, jj, kk) > self.sijr(ii, jj, rr)
            ]
            for ii in range(self._N)
            for jj in range(self._M)
        ]

        # Temporary check for duplicates
        for el in forbidden_idxs:
            if self.has_duplicates(el):
                print(
                    "**WARNING** Duplicates in preprocessed indices for constraint (4)"
                )

        # Update class instance attribute for constraint (4)
        self._preprocessed_constraint_idxs["C4"] = (True, forbidden_idxs)

    def preprocess_constraint_5(self):
        r"""
        Preprocesses constraint (5) and obtain a set
        of forbidden DTO and/or DLO indices to simplify
        the enforcing of the constraint when the QUBO
        mapping is used.
        This method directly changes the instance dictionary
        attribute :attr:`~MP4EOInsttance._preprocessed_constraint_idxs`.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        The set of forbidden DTO indices to enforce constraint
        (5) is defined as:

        .. math::

            \mathcal{G}_{i}
            =
            \left\{
                (s, j, k, k^\prime)
                \quad \vert \quad
                p_{is}^k > \lvert{t_{is}^k > t_{ij}^{k^\prime}}\rvert
            \right\}

        for all :math:`i = 1, \cdots, N`.
        """

        forbidden_idxs = [
            [
                (ss, jj, kk, kkp)
                for ss in range(self._M)
                for jj in range(self._M)
                for kk in range(self.thetaij(ii, ss))
                for kkp in (
                    range(kk + 1, self.thetaij(ii, jj))
                    if ss == jj
                    else range(self.thetaij(ii, jj))
                )
                if (
                    self.pijk(ii, ss, kk)
                    > abs(self.tijk(ii, ss, kk) - self.tijk(ii, jj, kkp))
                )
            ]
            for ii in range(self._N)
        ]

        # Temporary check for duplicates
        for ii in range(self._N):
            if self.has_duplicates(forbidden_idxs[ii]):
                print(
                    "**WARNING** Duplicates in preprocessed indices for constraint (5)"
                )

        # Update class instance attribute for constraint (5)
        self._preprocessed_constraint_idxs["C5"] = (True, forbidden_idxs)

    def preprocess_constraint_6(self):
        """
         Preprocesses constraint (6) and obtain a set
         of forbidden DTO and/or DLO indices to simplify
         the enforcing of the constraint when the QUBO
         mapping is used.
         This method directly changes the instance dictionary
         attribute :attr:`~MP4EOInsttance._preprocessed_constraint_idxs`.

         Arguments
         ---------
         :obj:`None`

         Returns
         -------
        `:obj:`None`

         Details
         -------
         The set of forbidden DTO indices to enforce constraint
         (6) is defined as:

         .. math::

             \mathcal{J}_{i}
             =
             \left\{
                 (g, j, r, r^\prime)
                 \quad \vert \quad
                 d_{ig}^r > \lvert{s_{ig}^r > s_{ij}^{r^\prime}}\rvert
             \right\}

         for all :math:`i = 1, \cdots, N`.
        """

        forbidden_idxs = [
            [
                (gg, jj, rr, rrp)
                for gg in range(self._M)
                for jj in range(self._M)
                for rr in range(self.sigma[ii])
                for rrp in (
                    range(rr + 1, self.sigma[ii])
                    if gg == jj
                    else range(self.sigma[ii])
                )
                if (
                    self.dijr(ii, gg, rr)
                    > abs(self.sijr(ii, gg, rr) - self.sijr(ii, jj, rrp))
                )
            ]
            for ii in range(self._N)
        ]

        # Temporary check for duplicates
        for ii in range(self._N):
            if self.has_duplicates(forbidden_idxs[ii]):
                print(
                    "**WARNING** Duplicates in preprocessed indices for constraint (6)"
                )

        # Updating class instance attribute for constraint (6)
        self._preprocessed_constraint_idxs["C6"] = (True, forbidden_idxs)

    def preprocess_constraint_7(self):
        """
         Preprocesses constraint (7) and obtain a set
         of allowed DTO and/or DLO indices to simplify
         the enforcing of the constraint when the QUBO
         mapping is used.
         This method directly changes the instance dictionary
         attribute :attr:`~MP4EOInsttance._preprocessed_constraint_idxs`.

         Arguments
         ---------
        `:obj:`None`

         Returns
         -------
         :obj:`None`

         Details
         -------
         The set of allowed DTO-DLO indices to enforce constraint
         (7) is defined as:

         .. math::

             \mathcal{L}_{ij}
             =
             \left\{
                 (k, r) \quad \vert \quad t_{ij}^k \leq s_{ij}^r
             \right\}

         for all :math:`i = 1, \cdots, N` and :math:`j = 1, \cdots, M`.
        """

        allowed_idxs = [
            [
                (kk, rr)
                for kk in range(self.thetaij(ii, jj))
                for rr in range(self.sigma[ii])
                if self.tijk(ii, jj, kk) <= self.sijr(ii, jj, rr)
            ]
            for ii in range(self._N)
            for jj in range(self._M)
        ]

        # Temporary check for duplicates
        for el in allowed_idxs:
            if self.has_duplicates(el):
                print(
                    "**WARNING** Duplicates in preprocessed indices for constraint (7)"
                )

        # Update class instance attribute for constraint (7)
        self._preprocessed_constraint_idxs["C7"] = (True, allowed_idxs)

        # Add slack variables
        self.update_n_slack_binaries()

    def get_preprocessed_c4_idxs(self, sat_idx, tar_idx):
        r"""
        Getter method for the list of indices in the preprocessed
        set related to constraint (4).

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the satellite
            in :math:`\mathcal{R}_{ij}`.

        tar_idx : int
            The index :math:`j` that identifies the target in
            :math:`\mathcal{R}_{ij}`.

        Returns
        -------
        forbidden_idxs : List[Tuple[int]]
            The list of forbidden indices :math:`\mathcal{R}_{ij}`.
            The output list can be an empty list even when constraint
            (4) is considered during the optimization.
        """

        assert (
            0 <= sat_idx < self._N
        ), f"Invalid index {sat_idx} for {self._N} satellites."
        assert (
            0 <= tar_idx < self._M
        ), f"Invalid index {tar_idx} for {self._M} targets."
        outlist = self._preprocessed_constraint_idxs["C4"][1]
        return outlist[sat_idx * self._M + tar_idx]

    def get_preprocessed_c5_idxs(self, sat_idx):
        r"""
        Getter method for the list of indices in the preprocessed
        set related to constraint (5).

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the satellite
            in :math:`\mathcal{G}_{i}`.

        Returns
        -------
        forbidden_idxs : List[Tuple[int]]
            The list of forbidden indices :math:`\mathcal{G}_{i}`.
            The output list can be an empty list even when constraint
            (5) is considered during the optimization.
        """

        assert (
            0 <= sat_idx < self._N
        ), f"Invalid index {sat_idx} for {self._N} satellites."
        return self._preprocessed_constraint_idxs["C5"][1][sat_idx]

    def get_preprocessed_c6_idxs(self, sat_idx):
        r"""
        Getter method for the list of indices in the preprocessed
        set related to constraint (6).

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the satellite
            in :math:`\mathcal{J}_{i}`.

        Returns
        -------

        forbidden_idxs : List[Tuple[int]]
            The list of forbidden indices :math:`\mathcal{J}_{i}`.
            The output list can be an empty list even when constraint
            (6) is considered during the optimization.
        """

        assert (
            0 <= sat_idx < self._N
        ), f"Invalid index {sat_idx} for {self._N} satellites."
        return self._preprocessed_constraint_idxs["C6"][1][sat_idx]

    def get_preprocessed_c7_idxs(self, sat_idx, tar_idx):
        r"""
        Getter method for the list of indices in the preprocessed
        set related to constraint (7).

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the satellite
            in :math:`\mathcal{L}_{ij}`.

        tar_idx : int
            The index :math:`j` that identifies the target in
            :math:`\mathcal{L}_{ij}`.

        Returns
        -------
        allowed_idxs : List[Tuple[int]]
            The list of allowed indices :math:`\mathcal{L}_{ij}`.
            The output list can be an empty list even when constraint
            (7) is considered during the optimization.
        """

        assert (
            0 <= sat_idx < self._N
        ), f"Invalid index {sat_idx} for {self._N} satellites."
        assert (
            0 <= tar_idx < self._M
        ), f"Invalid index {tar_idx} for {self._M} targets."
        outlist = self._preprocessed_constraint_idxs["C7"][1]
        return outlist[sat_idx * self._M + tar_idx]

    # ---------------------------------------------------------------------- #
    #                            Instance Methods                            #
    #                      **    QUBO Formulation   **                       #
    # ---------------------------------------------------------------------- #
    @cache
    def _dto_idxs_to_sg_idx(self, sat_idx, tar_idx, dto_idx):
        r"""
        Returns the site index in the spin-glass chain of a
        given mapped DTO binary :math:`x_{ijk}` identified by
        the index triplet :math:`\{i, j, k\}`. The output
        depends on the chosen :attr:`~MP4EOInstance._binaries_arrangement`.

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the
            satellite in the binary variable :math:`x_{ijk}`.

        tar_idx : int
            The index :math:`j` that identifies the
            target in the binary variable :math:`x_{ijk}`.

        dto_idx : int
            The index :math:`k` that identifies the
            data-take opportunity in the binary variable
            :math:`x_{ijk}`.

        Returns
        -------
        sg_idx : int
            The spin glass index :math:`\ell` that identifies
            the position of the spin variables :math:`\sigma_{\ell}`
            representing the DTO binary :math:`x_{ijk}` in the
            QUBO reformulation.
        """
        return self._dto_bin_sg_index_map[(sat_idx, tar_idx, dto_idx)]

    @cache
    def _dlo_idxs_to_sg_idx(self, sat_idx, tar_idx, dlo_idx):
        r"""
        Returns the site index in the spin-glass chain of a
        given mapped DLO binary :math:`y_{ijr}` identified by
        the index triplet :math:`\{i, j, r\}`. The output
        depends on the chosen :attr:`~MP4EOInstance._binaries_arrangement`.

        Arguments
        ---------
        sat_idx : int
            The index :math:`i` that identifies the
            satellite in the binary variable :math:`y_{ijr}`.

        tar_idx : int
            The index :math:`j` that identifies the
            target in the binary variable :math:`y_{ijr}`.

        dlo_idx : int
            The index :math:`r` that identifies the
            downlink opportunity in the binary variable
            :math:`y_{ijr}`.

        Returns
        -------
        sg_idx : int
            The spin glass index :math:`\ell` that identifies
            the position of the spin variables :math:`\sigma_{\ell}`
            representing the DLO binary :math:`y_{ijr}` in the
            QUBO reformulation.
        """
        return self._dlo_bin_sg_index_map[(sat_idx, tar_idx, dlo_idx)]

    @cache
    def _slack_idxs_to_sg_site(self, sat_idx, slack_idx):
        r"""
        Returns the site index in the spin-glass chain
        of a given mapped slack binary :math:`b_{i\alpha}`
        identified by the index pair :math:`\{i, \alpha\}`.
        The output depends on the chosen ``binaries_arrangement``.

        Arguments
        ---------

        sat_idx : int
            The index :math:`i` that identifies the
            satellite in the binary slack variable
            :math:`b_{i\alpha}`.

        slack_idx : int
            The index :math:`\alpha` that identifies
            the slack binary variable in :math:`b_{i\alpha}`.

        Returns
        -------
        sg_idx : int
            The spin glass index :math:`\ell` that identifies
            the position of the spin variables :math:`\sigma_{\ell}`
            representing the auxiliary slack binary :math:`b_{i\alpha}`
            in the QUBO reformulation.

        Details
        -------
        We currently choose to arrange all slack binaries at the end
        of the spin glass chain and at the end of the mission bitstring,
        regardless of the chosen arrangement for the DTO and DLO binaries.
        """
        try:
            return self._slack_bin_sg_index_map[(sat_idx, slack_idx)]
        except KeyError:
            raise MP4EOInstanceConstructionError(
                f"Required access to slack variables ({sat_idx}, {slack_idx}),"
                " but no such variable has been added to the problem."
            )

    def _build_binary_sg_index_mapping(self):
        """
        Pre-computes the mapping from DTO and DLO binary index to
        a specific one-dimensional arrangement of the binaries.
        The specific arrangement is selected through the chosen
        :attr:`~MP4EOInstance._binaries_arrangement`.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`
        """

        self._dto_bin_sg_index_map = {}
        self._dlo_bin_sg_index_map = {}
        sg_idx = 0  # Running index in the one-dimensional arrangement

        # Look-ups for efficiency
        thij = self.thetaij
        sigma = self.sigma

        # Arrange binaries along a one-dimensional chain

        if self._binaries_arrangement == "target-wise":
            for jj in range(self._M):
                for ii in range(self._N):
                    # DTO binaries x_{ijk}
                    for kk in range(thij(ii, jj)):
                        self._dto_bin_sg_index_map[(ii, jj, kk)] = sg_idx
                        sg_idx += 1

                    # DLO binaries y_{ijr}
                    for rr in range(sigma[ii]):
                        self._dlo_bin_sg_index_map[(ii, jj, rr)] = sg_idx
                        sg_idx += 1

        elif self._binaries_arrangement == "satellite-wise":
            for ii in range(self._N):
                for jj in range(self._M):
                    # DTO binaries x_{ijk}
                    for kk in range(thij(ii, jj)):
                        self._dto_bin_sg_index_map[(ii, jj, kk)] = sg_idx
                        sg_idx += 1

                    # DLO binaries y_{ijr}
                    for rr in range(sigma[ii]):
                        self._dlo_bin_sg_index_map[(ii, jj, rr)] = sg_idx
                        sg_idx += 1

        elif self._binaries_arrangement == "type-wise":
            # DTO binaries first
            for jj in range(self._M):
                for ii in range(self._N):
                    for kk in range(thij(ii, jj)):
                        self._dto_bin_sg_index_map[(ii, jj, kk)] = sg_idx
                        sg_idx += 1

            # Then DLO binaries
            for jj in range(self._M):
                for ii in range(self._N):
                    for rr in range(sigma[ii]):
                        self._dlo_bin_sg_index_map[(ii, jj, rr)] = sg_idx
                        sg_idx += 1

        else:
            raise NotImplementedError(
                f"Invalid binaries arrangement {self._binaries_arrangement}."
            )

    def split_decision_variable_bitstring(self, decision_variables_bitstring):
        """
        Splits the part related to the slack variables from
        the full decision variables bitstring of the MP4EO
        problem instance. If no slack variables where needed
        to enforce constraint (7), the splitting is trivial.

        Arguments
        ---------
        decision_variable_bitstring : np.ndarray[int]
            The bitstring that describe the configuration
            of all the decision variables involved in the
            MP4EO problem instance, both the MP4EO binaries
            and the eventually needed auxiliary binaries.

        Returns
        -------
        mission_bitstring : np.ndarray[int]
            The bitstring describing a mission without the part
            related to the auxiliary variables.

        slack_bitstring : np.ndarray[int]
            The bitstring describing the auxiliary (slack)
            variables configuration optimized to obtain the
            mission bitstring.
        """

        mission_bitstring = decision_variables_bitstring[
            : self.n_mp4eo_decision_variables
        ].astype(np.uint8)
        slack_bitstring = decision_variables_bitstring[
            self.n_mp4eo_decision_variables :
        ].astype(np.uint8)

        # Output
        return mission_bitstring, slack_bitstring

    # ---------------------------------------------------------------------- #
    #                            Instance Methods                            #
    #                        **    I/O routines   **                         #
    # ---------------------------------------------------------------------- #
    def save_instance(self, filename="mp4eo_instance", extension="pkl"):
        """
        Saves the MP4EO problem instance on a log file
        with a given ``extension``.

        Arguments
        ---------
        filename : str | pathlib.Path, optional
            Custom name for the instance file.
            The MP4EO instance ``filename`` will be stored
            in the :attr:`~MP4EOInstance.log_dir` repository.
            Default to ``mp4eo_instance``.

        extension : str, optional
            The extension of the output file.
            Default to pickle serialization ``pkl``.

        Returns
        -------
        :obj:`None`
        """

        filename += f"_{self.log_tag}.{extension}"
        instance_file = self.log_dir / filename

        if extension == "pkl":
            self._save_pickle(instance_file)
        else:
            raise NotImplementedError(
                f"Extension `{extension}` not implemented."
            )

    def _save_pickle(self, filename):
        """
        Serialize an MP4EO instance using :mod:`pickle`.

        Arguments
        ---------
        filename : str | pathlib.Path
            Full path to the file for logging the MP4EO instance.

        Returns
        -------
        :obj:`None`
        """

        filename = Path(filename)
        with filename.open("wb") as pf:
            pickle.dump(self, pf)

    def save_metrics(self, filename="mp4eo_instance_metrics"):
        """
        Saves instance metrics to a JSON file.

        Arguments
        ---------
        filename : str | pathlib.Path, optional
            Custom name for the instance metrics file.
            The log file ``filename`` will be stored
            in the :attr:`~MP4EOInstance.log_dir` repository.
            Default to ``mp4eo_instance_metrics``.

        Returns
        -------
        :obj:`None`
        """

        filename += f"_{self.log_tag}.json"
        metric_file = self.log_dir / filename

        with open(metric_file, "w") as fh:
            json.dump(self.metrics, fh, indent=4, separators=(", ", ": "))

    # ---------------------------------------------------------------------- #
    #                             Class Methods                              #
    # ---------------------------------------------------------------------- #
    @classmethod
    def from_random_distribution(
        cls,
        n_satellites,
        n_targets,
        instance_type="synthetic",
        binaries_arrangement="target-wise",
        constraints=None,
        ext_rng=None,
        parameter_ranges=None,
    ):
        """
        Constructs a random MP4EO problem instance by drawing parameters
        from the uniform probability distribution.

        Arguments
        ---------
        n_satellites : int
            The number of satellites in the constellation.

        n_targets : int
            The number of targets to be acquired within the mission.

        instance_type : str, optional
            Synthetic generation mode. The legacy name ``test`` is also
            accepted, but neither mode supplies default parameter ranges.

        parameter_ranges : dict
            Explicit, independently chosen inclusive integer bounds for all
            keys documented in ``get_synthetic_ranges``. No industrial ranges
            are bundled. Omitting this argument raises ValueError.

        binaries_arrangement : str, optional
            The order in which binary decision variables are
            arranged within the bitstring representing a generic
            planned mission.
            Available arrangement are as follows:

            - **satellite-wise** arranges all variables associated
            with each individual satellite in ascending target order,
            starting with the DTOs followed by the DLOs, where the
            satellite index is the slowest-varying parameter;
            - **target-wise** introduces all the binary variables associated
            with the same target, with the satellite index incrementing in
            ascending order, starting with the DTOs followed by the DLOs;
            - **type-wise** introduces all the DTOs variables first, with the
            target index varying more slowly than the satellite one;

            Default to ``target-wise``, i.e., the target index is the slowest
            varying one.

        constraints : Dict, optional
            Provides information about the constraints to be
            considered during optimization. Each key represents
            the name of a specific constraint ``C*``, while the
            corresponding value is a tuple. The first element of
            the tuple is a boolean flag indicating whether the
            constraint should be included. The second element is
            a float that specifies the weight of the constraint
            within the cost function.
            Default to :obj:`None`, i.e., free optimization.

        ext_rng : np.random.default_rng, optional
            A global random number generator from :mod:`numpy`
            that allows to produce different instance inside,
            for example, a for loop to obtain a bunch of mission
            planning problems.
            If :obj:`None`, the same instance is generated every
            time this constructor is called.
            Default to :obj:`None`.

        Returns
        -------
        obj : MP4EOInstance
            The generated random MP4EO problem instance.

        Details
        -------
        By default, the parameters for generating random MP4EO
        instances follow a uniform probability distribution.
        Random instances are meant for testing library and scripts.

        For random testing instances created by this constructor, all
        parameters are selected as integers, including those related to
        acquisition times and memory. If testing instances with real-valued
        parameters is required—such as in realistic use-case scenarios, this
        additional option can be easily introduced.
        """

        # Set random number generator for reproducibility
        rng = ext_rng or np.random.default_rng(0)
        iparam_ranges = cls._get_mp4eo_parameters_ranges(instance_type, parameter_ranges)

        # Initialize DTO parameters
        theta = rng.integers(
            *iparam_ranges["THETA"],
            endpoint=True,
            size=n_satellites * n_targets,
        )
        n_dto = np.sum(theta).item()
        dto_times = rng.integers(
            *iparam_ranges["DTO_TIME"], endpoint=True, size=n_dto
        )
        dto_preparation_times = rng.integers(
            *iparam_ranges["DTO_PREP_TIME"], endpoint=True, size=n_dto
        )

        # Initialize DLO parameters
        sigma = rng.integers(
            *iparam_ranges["SIGMA"], endpoint=True, size=n_satellites
        )
        n_dlo = n_targets * np.sum(sigma).item()
        dlo_times = rng.integers(
            *iparam_ranges["DLO_TIME"], endpoint=True, size=n_dlo
        )
        dlo_preparation_times = rng.integers(
            *iparam_ranges["DLO_PREP_TIME"],
            endpoint=True,
            size=n_dlo,
        )

        # Initialize onboard memory parameters
        required_target_memories = rng.integers(
            *iparam_ranges["MEM_SIZE"],
            endpoint=True,
            size=n_satellites * n_targets,
        )
        total_available_memories = rng.integers(
            *iparam_ranges["MEM_ONBOARD"],
            endpoint=True,
            size=n_satellites,
        )

        # Initialize cost function weights
        alpha = beta = 1
        if instance_type in ("synthetic", "test"):
            gamma = 0
            omega_dto = np.ones(n_dto)
            omega_dlo = np.ones(n_dlo)
        else:
            gamma = n_targets
            omega_dto = alpha * dto_times - gamma / n_targets
            omega_dlo = beta * dlo_times

        # Call the basic constructor
        obj = cls(
            n_satellites=n_satellites,
            n_targets=n_targets,
            theta=theta,
            n_dto=n_dto,
            dto_times=dto_times,
            dto_preparation_times=dto_preparation_times,
            sigma=sigma,
            n_dlo=n_dlo,
            dlo_times=dlo_times,
            dlo_preparation_times=dlo_preparation_times,
            required_target_memories=required_target_memories,
            total_available_memories=total_available_memories,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            omega_dto=omega_dto,
            omega_dlo=omega_dlo,
            iparam_range=iparam_ranges,
            binaries_arrangement=binaries_arrangement,
            constraints=constraints,
        )

        # Output
        return obj

    @classmethod
    def from_tasi_csv_file(*args, **kwargs):
        """Industrial data adapter excluded from the public synthetic release."""
        raise NotImplementedError(
            "Industrial CSV adapters are excluded; construct MP4EOInstance "
            "from independently specified synthetic arrays instead."
        )

    @classmethod
    def from_processed_tasi_csv_file(*args, **kwargs):
        """Industrial data adapter excluded from the public synthetic release."""
        raise NotImplementedError(
            "Industrial CSV adapters are excluded; construct MP4EOInstance "
            "from independently specified synthetic arrays instead."
        )

    @classmethod
    def from_custom_file(cls, filepath, extension="pkl"):
        """
        Reads the MP4EO problem instance from a standard
        file, e.g., by de-serializing a problem instance
        from a pickle file.

        Arguments
        ---------
        filepath : str | pathlib.Path
            The full path to the file where the serialized
            problem instance is stored.

        extension : str, optional
            The type of file to be parsed.
            Default to pickle files ``pkl``.

        Returns
        -------
        obj : MP4EOInstance
            The loaded MP4EO problem instance.
        """

        # Check the input file
        filepath = Path(filepath)
        if not filepath.exists() or filepath.suffix.lower() != f".{extension}":
            raise MP4EOInstanceConstructionError(
                f"Invalid input file {filepath}."
            )

        if extension == "pkl":
            with open(filepath, "rb") as instance_file:
                loaded_obj = pickle.load(instance_file)
        else:
            raise NotImplementedError(
                "Invalid from-file constructionn option."
            )

        # Ensure that the object loaded is indeed an MP4EO problem
        if not isinstance(loaded_obj, MP4EOInstance):
            raise MP4EOInstanceConstructionError(
                f"Expected object of type {cls.__name__},"
                f" got {type(loaded_obj)} instead."
            )

        # Re-initialize the instance
        obj = cls(
            n_satellites=loaded_obj.n_satellites,
            n_targets=loaded_obj.n_targets,
            theta=loaded_obj.theta,
            n_dto=loaded_obj.n_data_take_opportunity,
            dto_times=loaded_obj.dto_times,
            dto_preparation_times=loaded_obj.dto_preparation_times,
            sigma=loaded_obj.sigma,
            n_dlo=loaded_obj.n_downlink_opportunity,
            dlo_times=loaded_obj.dlo_times,
            dlo_preparation_times=loaded_obj.dlo_preparation_times,
            required_target_memories=loaded_obj.required_target_memories,
            total_available_memories=loaded_obj.total_available_memories,
            alpha=loaded_obj.alpha,
            beta=loaded_obj.beta,
            gamma=loaded_obj.gamma,
            omega_dto=loaded_obj.omega_dto,
            omega_dlo=loaded_obj.omega_dlo,
            iparam_range=loaded_obj.iparam_ranges,
            binaries_arrangement=loaded_obj.binaries_arrangement,
            constraints=loaded_obj.constraints,
        )

        # Output
        return obj
