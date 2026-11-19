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
MP4EO Mappers Submodule
=======================

This submodule implements various mappings of the constrained
Mission Planning for Earth Observation (MP4EO) linear integer
program, including both classical and quantum-inspired approaches.
"""


##############################################################################
#                                  MODULES                                   #
##############################################################################

# Standard library
from abc import ABC, abstractmethod
import json
import math
from pathlib import Path
from itertools import chain
import pickle
from collections import defaultdict

# Third-party libraries
import numpy as np
import scipy.sparse as ssp

# First-party libraries
from mp4eo_optimization.exceptions import (
    MP4EOQUBOMapperError,
    MP4EOQuditsMapperError,
)
from mp4eo_optimization.mp4eo_instance import *
from mp4eo_optimization.utils import ensure_init, embed_two_body



__all__ = ["BLIPMapper", "QUBOMapper", "QuditsMapper"]

##############################################################################
##############################################################################


# Public release modification (2026-11-19): retain research mapper; defer docplex import to BLIP construction.
class MP4EOMapper(ABC):
    """
    Base abstract class for an MP4EO mapper.
    Different mappers inherit from this base class and implement
    different classical and quantum-inspired mapping of the original
    optimization problem.

    Parameters
    ----------
    mp4eo_instance : mp4eo_optimization.MP4EOInstance
        The MP4EO problem instance to be mapped.

    mtype : str
        A name to identify the mapper.

    Attributes
    ----------
    _mp4eo_instance : mp4eo_optimization.MP4EOInstance
        The MP4EO problem instance to be mapped.

    _name : str
        The MP4EO mapper name.

    n_satellites : int
        The number of satellites in the input
        ``mp4eo_instance`` to be mapped.

    n_targets : int
        The number of EO targets in the input
        ``mp4eo_instance`` to be mapped.

    _primal_couplings_rescaling_factor : float
        The maximum value of the primal cost function couplings
        that can be used to rescale their order of magnitude for
        numerical stability.
        Not all the mappers needs this attribute.

    log_dir : str | Path
        The base directory for saving log files associated with the
        mapping procedure of an MP4EO problem instance.

    Raises
    ------
    TypeError
        If ``mp4eo_instance`` is not an instance of
        :class:`mp4eo_optimization.MP4EOInstance`.
        If ``mtype`` is not a :class:`str`.
    """

    # Define expected attributes for all mappers
    __slots__ = (
        "_mp4eo_instance",
        "_name",
        "n_satellites",
        "n_targets",
        "log_dir",
    )

    # ---------------------------------------------------------------------- #
    #                              Constructor                               #
    # ---------------------------------------------------------------------- #
    def __init__(self, mp4eo_instance, mtype):
        """Initializes a generic MP4EO mapper API."""

        # Check instance parameters
        if not isinstance(mp4eo_instance, MP4EOInstance):
            raise TypeError(
                "Expected an instance of MP4EOInstance,"
                f" but got {type(mp4eo_instance).__name__}"
            )
        if not isinstance(mtype, str):
            raise TypeError(
                f"Expected a string, but got {type(mtype).__name__}"
            )

        # Initialize attributes
        self._mp4eo_instance = mp4eo_instance
        self._name = mtype
        self.log_dir = Path(f"MP4EO_RESULTS/mapping/")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Define order of magnitude for the original problem
        primal_couplings = np.concatenate(
            (self._mp4eo_instance.omega_dto, self._mp4eo_instance.omega_dlo)
        )
        self._primal_couplings_rescaling_factor = np.max(
            np.abs(primal_couplings)
        )

        # Avoid repeated lookups
        self.n_satellites = self._mp4eo_instance.n_satellites
        self.n_targets = self._mp4eo_instance.n_targets

    # ---------------------------------------------------------------------- #
    #                               Properties                               #
    # ---------------------------------------------------------------------- #
    @property
    def mp4eo_instance(self):
        """Read-only access to the MP4EO problem instance to be mapped."""
        return self._mp4eo_instance

    @property
    def mapping_type(self):
        """Read-only access to the mapper name."""
        return self._name

    # ---------------------------------------------------------------------- #
    #                            Abstract Methods                            #
    # ---------------------------------------------------------------------- #
    @property
    @abstractmethod
    def get_mspace_dim(self):
        """Return the total dimension of the mapped problem."""
        pass

    @abstractmethod
    def map(self, *args, **kwargs):
        """
        Abstract method to perform the mapping of the MP4EO problem instance.

        Arguments
        ---------

        *args : Tuple
            Optional positional arguments depending on the specific
            mapper.

        **kwargs : Dict
            Optional keyword arguments for specific mapping needs.
        """
        pass


class BLIPMapper(MP4EOMapper):
    """
    Mapping an MP4EO instance to its corresponding Binary
    Linear Integer Program (BLIP).

    Parameters
    ----------
    mp4eo_instance : mp4eo_optimization.MP4EOInstance
        The MP4EO problem instance to be mapped into
        a BLIP problem.

    Attributes
    ----------
    Same as the base class :class:`~MP4EOMapper`

    _blip_model : docplex.mp.model.Model
        The BLIP formulation of the ``mp4eo_instance``.

    Details
    -------
    The BLIP formulation of the given MP4EO problem instance is
    constructed via the :mod:`docplex.mp` API.
    This Python module is supported by IBM's CPLEX Optimization
    Studio and use mathematical programming to obtain solutions
    to combinatorial optimization problems.
    More details can be found in the reference manual
    `<https://ibmdecisionoptimization.github.io/docplex-doc/mp/refman.html>`_.
    """

    # ---------------------------------------------------------------------- #
    #                              Constructor                               #
    # ---------------------------------------------------------------------- #
    def __init__(self, mp4eo_instance):
        """Initializes the Binary Linear Integer Program (BLIP) mapper."""
        super().__init__(mp4eo_instance, "BLIP Mapper")
        self.log_dir = self.log_dir / "blip/"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Specific attributes
        try:
            from docplex.mp.model import Model
        except ImportError as exc:
            raise ImportError(
                "BLIPMapper requires the optional 'docplex' dependency."
            ) from exc

        self._blip_model = Model(name="MP4EO BLIP Model")

    # ---------------------------------------------------------------------- #
    #                               Properties                               #
    # ---------------------------------------------------------------------- #
    @property
    def blip_model(self):
        """Read-only access to the BLIP model."""
        return self._blip_model

    @property
    def get_mspace_dim(self):
        """
        Returns the total number of decision variables
        in the Binary Linear Integer Program.
        """
        return self._mp4eo_instance.n_mp4eo_decision_variables

    # ---------------------------------------------------------------------- #
    #                            Instance Methods                            #
    # ---------------------------------------------------------------------- #
    def prettyprint(self):
        """
        Prints complete information about the mapped Binary Linear Integer
        Program.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`
        """

        # Heading
        head = "MP4EO Binary Linear Integer Program details"
        sep_line = "-" * len(head)
        print(f"{sep_line}\n{head}\n{sep_line}\n")

        # Call to DOcplex Python API method
        print(self._blip_model.prettyprint())
        print("\n\n\n")

    def _add_decision_variables(self):
        """
        Adds the decision variables defining the BLIP model.
        Uses the pre-computed mappings for the one-dimensional
        bit-string arrangement of decision variables at the
        :class:`mp4eo_optimization.mp4eo_instance.MP4EOInstance`
        level.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`
        """

        # Retrieve correct arrangement of decision variables
        all_vars = sorted(
            [
                (idx, f"x_{ii}{jj}{kk}")
                for (
                    ii,
                    jj,
                    kk,
                ), idx in self._mp4eo_instance.dto_bin_sg_index_map.items()
            ]
            + [
                (idx, f"y_{ii}{jj}{rr}")
                for (
                    ii,
                    jj,
                    rr,
                ), idx in self._mp4eo_instance.dlo_bin_sg_index_map.items()
            ],
            key=lambda x: x[0],
        )

        # Add decision variables to the model
        for _, var_name in all_vars:
            self._blip_model.binary_var(var_name)

    def _add_constraint_1(self):
        """
        Adds mission constraint (1) to the MP4EO Binary Linear Integer
        Program.
        Adds nothing if constraint (1) is not considered for optimization.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        See :class:`mp4eo_optimization.mp4eo_instance.MP4EOInstance` for
        details on constraint (1).
        """

        if not self._mp4eo_instance.get_constraint("C1")[0]:
            return

        for jj in range(self.n_targets):
            lhs = [
                self._blip_model.get_var_by_name(f"x_{ii}{jj}{kk}")
                for ii in range(self.n_satellites)
                for kk in range(self._mp4eo_instance.thetaij(ii, jj))
            ]

            self._blip_model.add_constraint(
                ct=self._blip_model.sum(lhs) == 1,
                ctname=f"(1) One acquisition attempt per target j = {jj}",
            )

    def _add_constraint_2(self):
        """
        Adds mission constraint (2) to the MP4EO Binary Linear Integer
        Program.
        Adds nothing if constraint (2) is not considered for optimization.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        See :class:`mp4eo_optimization.mp4eo_instance.MP4EOInstance` for
        details on constraint (2).
        """

        if not self._mp4eo_instance.get_constraint("C2")[0]:
            return

        for jj in range(self.n_targets):
            lhs = [
                self._blip_model.get_var_by_name(f"y_{ii}{jj}{rr}")
                for ii in range(self.n_satellites)
                for rr in range(self._mp4eo_instance.sigma[ii])
            ]

            self._blip_model.add_constraint(
                ct=self._blip_model.sum(lhs) == 1,
                ctname=f"(2) One download attempt per target j = {jj}",
            )

    def _add_constraint_3(self):
        """
        Adds mission constraint (3) to the MP4EO Binary Linear Integer
        Program.
        Adds nothing if constraint (3) is not considered for optimization.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        See :class:`mp4eo_optimization.mp4eo_instance.MP4EOInstance` for
        details on constraint (3).
        """

        if not self._mp4eo_instance.get_constraint("C3")[0]:
            return

        for ii in range(self.n_satellites):
            for jj in range(self.n_targets):
                dto_terms = [
                    (self._blip_model.get_var_by_name(f"x_{ii}{jj}{kk}"), 1)
                    for kk in range(self._mp4eo_instance.thetaij(ii, jj))
                ]
                dlo_terms = [
                    (self._blip_model.get_var_by_name(f"y_{ii}{jj}{rr}"), -1)
                    for rr in range(self._mp4eo_instance.sigma[ii])
                ]

                terms = dto_terms + dlo_terms
                constraint_expr = self._blip_model.linear_expr(dict(terms))

                cname = (
                    f"(3) Acquired target j = {jj} must be "
                    f"downloaded from satellite i = {ii}"
                )

                self._blip_model.add_constraint(
                    ct=constraint_expr == 0, ctname=cname
                )

    def _add_constraint_4(self):
        """
        Adds mission constraint (4) to the MP4EO Binary Linear Integer
        Program.
        Adds nothing if constraint (4) is not considered for optimization.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        See :class:`mp4eo_optimization.mp4eo_instance.MP4EOInstance` for
        details on constraint (4).
        """

        if not self._mp4eo_instance.get_preprocessed_constraint_flag("C4"):
            return

        for ii in range(self.n_satellites):
            for jj in range(self.n_targets):
                Fij = self._mp4eo_instance.get_preprocessed_c4_idxs(ii, jj)

                for kk, rr in Fij:
                    dto_binary = self._blip_model.get_var_by_name(
                        f"x_{ii}{jj}{kk}"
                    )
                    dlo_binary = self._blip_model.get_var_by_name(
                        f"y_{ii}{jj}{rr}"
                    )

                    cname = (
                        f"(4) Target j = {jj} acquisition in k = {kk} precedes"
                        f" its download in r = {rr} for satellite i = {ii}"
                    )

                    self._blip_model.add_constraint(
                        ct=dto_binary + dlo_binary <= 1, ctname=cname
                    )

    def _add_constraint_5(self):
        """
        Adds mission constraint (5) to the MP4EO Binary Linear Integer
        Program.
        Adds nothing if constraint (5) is not considered for optimization.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        See :class:`mp4eo_optimization.mp4eo_instance.MP4EOInstance` for
        details on constraint (5).
        """

        if not self._mp4eo_instance.get_preprocessed_constraint_flag("C5"):
            return

        for ii in range(self.n_satellites):
            Gi = self._mp4eo_instance.get_preprocessed_c5_idxs(ii)

            for gg, jj, kk, kkp in Gi:
                x1 = self._blip_model.get_var_by_name(f"x_{ii}{gg}{kk}")
                x2 = self._blip_model.get_var_by_name(f"x_{ii}{jj}{kkp}")

                cname = (
                    f"(5) Satellite i = {ii} must have enough preparation "
                    f"time to acquire targets g == {gg} and j = {jj} at "
                    f"opportunities k = {kk} and k' = {kkp}"
                )

                self._blip_model.add_constraint(ct=x1 + x2 <= 1, ctname=cname)

    def _add_constraint_6(self):
        """
        Adds mission constraint (6) to the MP4EO Binary Linear Integer
        Program.
        Adds nothing if constraint (6) is not considered for optimization.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        See :class:`mp4eo_optimization.mp4eo_instance.MP4EOInstance` for
        details on constraint (6).
        """

        if not self._mp4eo_instance.get_preprocessed_constraint_flag("C6"):
            return

        for ii in range(self.n_satellites):
            Ki = self._mp4eo_instance.get_preprocessed_c6_idxs(ii)

            for gg, jj, rr, rrp in Ki:
                y1 = self._blip_model.get_var_by_name(f"y_{ii}{gg}{rr}")
                y2 = self._blip_model.get_var_by_name(f"y_{ii}{jj}{rrp}")

                cname = (
                    f"(6) Satellite i = {ii} must have enough preparation "
                    f"time to download targets g == {gg} and j = {jj} at "
                    f"opportunities r = {rr} and r' = {rrp}"
                )

                self._blip_model.add_constraint(ct=y1 + y2 <= 1, ctname=cname)

    def _add_constraint_7(self):
        """
        Adds mission constraint (7) to the MP4EO Binary Linear Integer
        Program.
        Adds nothing if constraint (7) is not considered for optimization.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        See :class:`mp4eo_optimization.mp4eo_instance.MP4EOInstance` for
        details on constraint (7).
        """

        if not self._mp4eo_instance.get_preprocessed_constraint_flag("C7"):
            return

        for ii in range(self.n_satellites):
            constraint = self._blip_model.linear_expr()
            qi_tot = self._mp4eo_instance.total_available_memories[ii]

            for jj in range(self.n_targets):
                Lij = self._mp4eo_instance.get_preprocessed_c7_idxs(ii, jj)
                qij = self._mp4eo_instance.qij(ii, jj)

                for kk, rr in Lij:
                    constraint += qij * self._blip_model.get_var_by_name(
                        f"x_{ii}{jj}{kk}"
                    )
                    constraint -= qij * self._blip_model.get_var_by_name(
                        f"x_{ii}{jj}{rr}"
                    )

            self._blip_model.add_constraint(
                ct=constraint <= qi_tot,
                ctname=f"(7) Available memory onboard of satellite {ii}",
            )

    def map(self):
        """
        Maps the MP4EO problem instance onto the corresponding
        Binary Linear Integer Program (BLIP).

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`
        """

        # Add decision variables depending on the chosen bitstring order
        self._add_decision_variables()

        # Define look-ups for efficient access
        get_var = self._blip_model.get_var_by_name
        wijk = self._mp4eo_instance.wijk
        wijr = self._mp4eo_instance.wijr
        thij = self._mp4eo_instance.thetaij
        sigma = self._mp4eo_instance.sigma

        # Define the BLIP cost function
        blip_objective = self._blip_model.sum(
            wijk(ii, jj, kk) * get_var(f"x_{ii}{jj}{kk}")
            for ii in range(self.n_satellites)
            for jj in range(self.n_targets)
            for kk in range(thij(ii, jj))
        ) + self._blip_model.sum(
            wijr(ii, jj, rr) * get_var(f"y_{ii}{jj}{rr}")
            for ii in range(self.n_satellites)
            for jj in range(self.n_targets)
            for rr in range(sigma[ii])
        )
        self._blip_model.minimize(blip_objective)

        # Add linear equality constraints
        self._add_constraint_1()
        self._add_constraint_2()
        self._add_constraint_3()

        # Add chosen linear inequality constraints
        self._add_constraint_4()
        self._add_constraint_5()
        self._add_constraint_6()
        self._add_constraint_7()


class QUBOMapper(MP4EOMapper):
    """
    Mapping an MP4EO instance to its corresponding Quadratic
    Unconstrained Binary Optimization (QUBO) problem.

    Parameters
    ----------
    mp4eo_instance : mp4eo_optimization.mp4eo_instance.MP4EOInstance
        The MP4EO problem instance to be mapped into
        a QUBO problem.

    Attributes
    ----------
    Same as the base class :class:`~MP4EOMapper`

    _offset : float
        The QUBO (energy) offset.

    _qubo_matrix : np.ndarray[float]
        The QUBO matrix.

    _spinglass_couplings : Dict[str, float | np.ndarray[float]]
        The dictionary of the couplings defining the spin
        glass model related to the the mapped QUBO problem.

    _penalties_cte : Dict[str, float]
        The penalty constants :math:`\lambda_i \in \mathbb{R}`
        used to enforce constraints as penalty terms and
        construct the mapped QUBO problem.
        Keys are constraint names, values are the value of the
        penalty constants :math:`\lambda`.

    Details
    -------
    MP4EO constraints are incorporated into the cost function as
    penalty terms, as is typically done in QUBOs.
    During pre-processing of the input ``mp4eo_instance`` parameters,
    we minimize the number of auxiliary variables needed to enforce
    inequality constraints.
    See :class:`mp4eo_optimization.MP4EOInstance` for more details
    on problem constraints.

    The cost function of a generic QUBO problem can be written as

    .. math::

        \mathcal{F}_{\mathrm{QUBO}}
        =
        Cte + \sum_{j=1}^{n_Q} \mathcal{Q}_{jj} z_j +
        \sum_{j < k}^{n_Q} \mathcal{Q}_{jk} z_j z_k

    where :math:`Cte` is the so-called (energy) offset,
    :math:`n_Q` is the number of binary decision variables
    of the QUBO problem, and :math:`\mathcal{Q}` is the
    :math:`n_Q \times n_Q` real-valued symmetric matrix that
    completely characterizes the QUBO problem.

    Notes
    -----
    This mapping is required by various solvers, including those
    based on tensor-network methods when the original MP4EO problem
    is encoded with qubits.

    When it comes to the penalty constants
    :math:`\lambda_{\ell} \in \mathbb{R}`, with :math:`\ell = 1, \cdots, 7`,
    needed to construct the QUBO and enforce constraints, we use a
    default strategy during the initialization of the mapper object.
    This strategy sets the :math:`\lambda` values for the constraints
    based on a heuristic, which may not always be effective.

    If you prefer not to use the default, the :math:`lambda`s can be
    manually tuned through the mapping method :meth:`~QUBOMapper.map`
    using two different strategies.
    Alternatively, you can directly provide the values.

    Solvers based on the QUBO reformulation can also auto-tune the
    :math:`lambda`s through additional on-the-fly optimization.
    See :mod:`mp4eo_optimization.mp4eo_solvers` for more details.
    """

    # ---------------------------------------------------------------------- #
    #                              Constructor                               #
    # ---------------------------------------------------------------------- #
    def __init__(self, mp4eo_instance):
        """Initializes the QUBO mapper."""

        # Inherit base-class constructor
        super().__init__(mp4eo_instance, "QUBO Mapper")
        self.log_dir = self.log_dir / "qubo/"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._offset = self._qubo_matrix = self._spinglass_couplings = None

        self._set_qubo_penalty_constants()

    # ---------------------------------------------------------------------- #
    #                               Properties                               #
    # ---------------------------------------------------------------------- #
    @property
    def qubo_penalties(self):
        """Read-only access to QUBO penalties values."""
        return self._penalties_cte

    # Dynamically create properties for the QUBO penalty constants
    def _make_lambda_property(index):
        cname = f"C{index}"

        def getter(self):
            if cname not in self._penalties_cte:
                raise AttributeError(f"Penalty constant '{cname}' not found.")
            return self._penalties_cte[cname]

        def setter(self, new_lambda):
            if new_lambda is not None:
                if not isinstance(new_lambda, (int, float)):
                    raise ValueError(
                        "QUBO penalty constant must be a numerical value."
                    )
                self._penalties_cte[cname] = new_lambda

        return property(getter, setter)

    for ii in range(1, 8):
        locals()[f"lambda{ii}"] = _make_lambda_property(ii)
    del ii

    @property
    def get_mspace_dim(self):
        """
        Returns the total number of decision variables
        in the QUBO problem, i.e., the number of spins
        in the corresponding spin glass system.
        """
        return self._mp4eo_instance.n_decision_variables

    @property
    @ensure_init("_offset")
    def offset(self):
        """Read-only access to the QUBO offset :math:`Cte`."""
        return self._offset

    @property
    @ensure_init("_qubo_matrix")
    def qubo_matrix(self):
        """Read-only access to the QUBO matrix :math:`\mathcal{Q}`."""
        return self._qubo_matrix

    @property
    @ensure_init("_spinglass_couplings")
    def spinglass_couplings(self):
        """Read-only access to the spin glass couplings."""
        return self._spinglass_couplings

    @property
    @ensure_init("_spinglass_couplings")
    def spinglass_shift(self):
        """Read-only access to the spin glass energy offset."""
        return self._spinglass_couplings["offset"]

    @property
    @ensure_init("_spinglass_couplings")
    def spinglass_hfields(self):
        """Read-only access to the spin glass biases."""
        return self._spinglass_couplings["one-body"]

    @property
    @ensure_init("_spinglass_couplings")
    def spinglass_interactions(self):
        """
        Read-only access to the spin glass two-body
        interaction strengths.
        """
        return self._spinglass_couplings["two-body"]

    # ---------------------------------------------------------------------- #
    #                            Instance Methods                            #
    # ---------------------------------------------------------------------- #
    def map(self, **kwargs):
        """
        Maps the MP4EO problem instance onto the corresponding
        QUBO problem.

        Arguments
        ---------
        **kwargs : Dict[str, Any]
            Additional optional arguments, e.g., to change values
            to the QUBO penalty coonstants before constructing the
            QUBO matrix, that could be useful when reading a stored
            MP4EO instance from an old file.

            To tune the :math:`\lambda`s based on a heuristic related
            to the couplings of the cost function for the primal
            optimization problem, use the multiplicative penalty ratios
            as optional arguments:

            - ``lambda1_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_1` related to constraint (1).
                Default to :obj:`None`.

            - ``lambda2_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_2` related to constraint (2).
                Default to :obj:`None`.

            - ``lambda3_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_3` related to constraint (3).
                Default to :obj:`None`.

            - ``lambda4_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_4` related to constraint (4).
                Default to :obj:`None`.

            - ``lambda5_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_5` related to constraint (5).
                Default to :obj:`None`.

            - ``lambda6_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_6` related to constraint (6).
                Default to :obj:`None`.

            - ``lambda7_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_7` related to constraint (7).
                Default to :obj:`None`.

            In this case, penalty constants are defined as

            .. math::

            \lambda_{\ell}
            =
            \bar{\lambda}_{\ell}
            \mathcal{C}\left(\boldsymbol{\omega}\right)

            where :math:`\bar{\lambda}_{\ell} \in \mathbb{R}` is the ratio
            that can be tuned with the above “ratio” arguments, while
            :math:`\mathcal{C}\left(\boldsymbol{\omega}\right) \in \mathbb{R}`
            is some average related to the primal problem couplings
            :math:`\boldsymbol{\omega}` that defines the cost function.
            The way this last quantity is computed can be set via the additional
            optional arguments:

            - ``baseline``: str
                Use `mean` to set :math:`\mathcal{C}` as the mean value of
                the primal cost function couplings;
                `max` to use as baseline for the tuning of the QUBO penalty
                constants the maximum value of the couplings.
                The default is `mean`.

            Alternatively, to directly set the penalty constant values, use the
            following optional arguments:

            - ``lambda1`` : float
                New value for the QUBO penalty constant :math:`\lambda_1`
                related to constraint (1).
                Default to :obj:`None`.

            - ``lambda2`` : float
                New value for the QUBO penalty constant :math:`\lambda_2`
                related to constraint (2).
                Default to :obj:`None`.

            - ``lambda3`` : float
                New value for the QUBO penalty constant :math:`\lambda_3`
                related to constraint (3).
                Default to :obj:`None`.

            - ``lambda4`` : float
                New value for the QUBO penalty constant :math:`\lambda_4`
                related to constraint (4).
                Default to :obj:`None`.

            - ``lambda5`` : float
                New value for the QUBO penalty constant :math:`\lambda_5`
                related to constraint (5).
                Default to :obj:`None`.

            - ``lambda6`` : float
                New value for the QUBO penalty constant :math:`\lambda_6`
                related to constraint (6).
                Default to :obj:`None`.

            - ``lambda7`` : float
                New value for the QUBO penalty constant :math:`\lambda_7`
                related to constraint (7).
                Default to :obj:`None`.

            If you need to rescale the couplings of the primal optimization
            problem cost function for numerical stability, you can provide
            an optional boolen argument ``rescale_primal_couplings``, whose
            default is set to :obj:`True`.

        Returns
        -------
        :obj:`None`
        """

        # Set the QUBO penalty constants
        self._set_qubo_penalty_constants(**kwargs)

        # Compute QUBO cost function offset
        self._compute_qubo_offset()

        # Construct QUBO matrix
        self.construct_qubo_matrix(
            kwargs.get("rescale_primal_couplings", True)
        )

    def _compute_qubo_offset(self):
        """Computes and returns the (energy) offset :math:`Cte`."""

        tot_memories = self._mp4eo_instance.total_available_memories

        self._offset = self.n_targets * (self.lambda1 + self.lambda2)
        self._offset += self.lambda7 * np.sum(np.square(tot_memories))

    def _compute_dto_qubo_biases(self, rescale_primal_couplings=True):
        """
        Computes and returns the DTO-binary biases.

        Arguments
        ---------
        rescale_primal_couplings : bool, optional
            Whether to rescale the primal couplings that
            define the cost function of the primal optimization
            problem to compute QUBO matrix elements within a
            more numerically stable framework.
            Default to :obj:`True`, i.e., the primal couplings
            will be rescaled.

        Returns
        -------
        :obj:`None`
        """

        dto_biases = {}
        cte_bias = self.lambda3 - self.lambda1
        tot_memories = self._mp4eo_instance.total_available_memories
        if rescale_primal_couplings:
            rescale_factor = self._primal_couplings_rescaling_factor
        else:
            rescale_factor = 1

        for ii in range(self.n_satellites):
            qi = tot_memories[ii]

            for jj in range(self.n_targets):
                qij = self._mp4eo_instance.qij(ii, jj)
                thij = self._mp4eo_instance.thetaij(ii, jj)
                mem_bias = qi * qij

                for kk in range(thij):
                    qubo_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                        sat_idx=ii, tar_idx=jj, dto_idx=kk
                    )
                    dto_biases[qubo_idx] = (
                        dto_biases.get(qubo_idx, 0)
                        + self._mp4eo_instance.wijk(
                            sat_idx=ii, tar_idx=jj, dto_idx=kk
                        )
                        / rescale_factor
                        + cte_bias
                    )

                if self._mp4eo_instance.get_preprocessed_constraint_flag("C7"):
                    Lij = self._mp4eo_instance.get_preprocessed_c7_idxs(
                        sat_idx=ii, tar_idx=jj
                    )
                    for kk, _ in Lij:
                        qubo_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dto_idx=kk
                        )
                        dto_biases[qubo_idx] = (
                            dto_biases.get(qubo_idx, 0)
                            - 2 * self.lambda7 * mem_bias
                        )

        # Output
        return dto_biases

    def _compute_dlo_qubo_biases(self, rescale_primal_couplings=True):
        """
        Computes and returns the DLO-binary biases.

        Arguments
        ---------
        rescale_primal_couplings : bool, optional
            Whether to rescale the primal couplings that
            define the cost function of the primal optimization
            problem to compute QUBO matrix elements within a
            more numerically stable framework.
            Default to :obj:`True`, i.e., the primal couplings
            will be rescaled.

        Returns
        -------
        :obj:`None`
        """

        dlo_biases = {}
        cte_bias = self.lambda3 - self.lambda2
        tot_memories = self._mp4eo_instance.total_available_memories
        sigma = self._mp4eo_instance.sigma
        if rescale_primal_couplings:
            rescale_factor = self._primal_couplings_rescaling_factor
        else:
            rescale_factor = 1

        for ii in range(self.n_satellites):
            qi = tot_memories[ii]
            si = sigma[ii]

            for jj in range(self.n_targets):
                qij = self._mp4eo_instance.qij(ii, jj)
                mem_bias = qi * qij

                for rr in range(si):
                    qubo_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                        sat_idx=ii, tar_idx=jj, dlo_idx=rr
                    )
                    dlo_biases[qubo_idx] = (
                        dlo_biases.get(qubo_idx, 0)
                        + self._mp4eo_instance.wijr(
                            sat_idx=ii, tar_idx=jj, dlo_idx=rr
                        )
                        / rescale_factor
                        + cte_bias
                    )

                if self._mp4eo_instance.get_preprocessed_constraint_flag("C7"):
                    Lij = self._mp4eo_instance.get_preprocessed_c7_idxs(
                        sat_idx=ii, tar_idx=jj
                    )
                    for _, rr in Lij:
                        qubo_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dlo_idx=rr
                        )
                        dlo_biases[qubo_idx] = (
                            dlo_biases.get(qubo_idx, 0)
                            + 2 * self.lambda7 * mem_bias
                        )

        # Output
        return dlo_biases

    def _compute_slack_qubo_biases(self):
        """Computes and returns the slack-binary biases."""

        slack_biases = {}
        tot_memories = self._mp4eo_instance.total_available_memories

        for ii in range(self.n_satellites):
            qi = tot_memories[ii]

            for aa in range(self._mp4eo_instance.nslacki(ii)):
                qubo_idx = self._mp4eo_instance._slack_idxs_to_sg_site(
                    sat_idx=ii, slack_idx=aa
                )
                slack_biases[qubo_idx] = slack_biases.get(
                    qubo_idx, 0
                ) + 2 * self.lambda7 * (2**aa * (qi + 2 ** (aa - 1)))

        # Output
        return slack_biases

    def _compute_dto_dto_qubo_couplings(self):
        """Computes and returns the DTO-DTO binary-interaction couplings."""

        dto_dto_couplings = {}
        cte_13 = 2 * self.lambda1 + self.lambda3

        for ii in range(self.n_satellites):
            for jj in range(self.n_targets):
                thij = self._mp4eo_instance.thetaij(ii, jj)
                for kk in range(thij):
                    for kkp in range(kk + 1, thij):
                        row_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dto_idx=kk
                        )
                        col_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dto_idx=kkp
                        )
                        qubo_key = (row_idx, col_idx)
                        dto_dto_couplings[qubo_key] = (
                            dto_dto_couplings.get(qubo_key, 0) + cte_13
                        )

        for ii in range(self.n_satellites):
            for iip in range(ii + 1, self.n_satellites):
                for jj in range(self.n_targets):
                    thij = self._mp4eo_instance.thetaij(ii, jj)
                    thipj = self._mp4eo_instance.thetaij(iip, jj)
                    for kk in range(thij):
                        row_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dto_idx=kk
                        )
                        for kkp in range(thipj):
                            col_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                                sat_idx=iip, tar_idx=jj, dto_idx=kkp
                            )
                            qubo_key = (row_idx, col_idx)
                            dto_dto_couplings[qubo_key] = (
                                dto_dto_couplings.get(qubo_key, 0)
                                + 2 * self.lambda1
                            )

        if self._mp4eo_instance.get_preprocessed_constraint_flag("C5"):
            for ii in range(self.n_satellites):
                Gi = self._mp4eo_instance.get_preprocessed_c5_idxs(ii)
                for gg, jj, kk, kkp in Gi:
                    row_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                        sat_idx=ii, tar_idx=gg, dto_idx=kk
                    )
                    col_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                        sat_idx=ii, tar_idx=jj, dto_idx=kkp
                    )
                    qubo_key = (row_idx, col_idx)
                    dto_dto_couplings[qubo_key] = (
                        dto_dto_couplings.get(qubo_key, 0) + self.lambda5
                    )

        if self._mp4eo_instance.get_preprocessed_constraint_flag("C7"):
            for ii in range(self.n_satellites):
                for jj in range(self.n_targets):
                    qij = self._mp4eo_instance.qij(ii, jj)
                    Lij = self._mp4eo_instance.get_preprocessed_c7_idxs(
                        sat_idx=ii, tar_idx=jj
                    )

                    for jjp in range(self.n_targets):
                        qijp = self._mp4eo_instance.qij(ii, jjp)
                        Lijp = self._mp4eo_instance.get_preprocessed_c7_idxs(
                            ii, jjp
                        )
                        cte_7qq = self.lambda7 * qij * qijp
                        for kk, _ in Lij:
                            row_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                                ii, jj, kk
                            )

                            for kkp, _ in Lijp:
                                col_idx = (
                                    self._mp4eo_instance._dto_idxs_to_sg_idx(
                                        ii, jjp, kkp
                                    )
                                )
                                qubo_key = (row_idx, col_idx)
                                dto_dto_couplings[qubo_key] = (
                                    dto_dto_couplings.get(qubo_key, 0)
                                    + cte_7qq
                                )

        # Output
        return dto_dto_couplings

    def _compute_dlo_dlo_qubo_couplings(self):
        """Computes and returns the DLO-DLO binary-interaction couplings."""

        dlo_dlo_couplings = {}
        cte_23 = 2 * (self.lambda2 + self.lambda3)
        sigma = self._mp4eo_instance.sigma

        for ii in range(self.n_satellites):
            si = sigma[ii]
            for jj in range(self.n_targets):
                for rr in range(si):
                    for rrp in range(rr + 1, si):
                        row_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dlo_idx=rr
                        )
                        col_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dlo_idx=rrp
                        )
                        qubo_key = (row_idx, col_idx)
                        dlo_dlo_couplings[qubo_key] = (
                            dlo_dlo_couplings.get(qubo_key, 0) + cte_23
                        )

        for ii in range(self.n_satellites):
            si = sigma[ii]
            for iip in range(ii + 1, self.n_satellites):
                sip = sigma[iip]
                for jj in range(self.n_targets):
                    for rr in range(si):
                        row_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dlo_idx=rr
                        )
                        for rrp in range(sip):
                            col_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                                sat_idx=iip, tar_idx=jj, dlo_idx=rrp
                            )
                            qubo_key = (row_idx, col_idx)
                            dlo_dlo_couplings[qubo_key] = (
                                dlo_dlo_couplings.get(qubo_key, 0)
                                + 2 * self.lambda2
                            )

        if self._mp4eo_instance.get_preprocessed_constraint_flag("C6"):
            for ii in range(self.n_satellites):
                Ki = self._mp4eo_instance.get_preprocessed_c6_idxs(ii)
                for gg, jj, rr, rrp in Ki:
                    row_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                        sat_idx=ii, tar_idx=gg, dlo_idx=rr
                    )
                    col_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                        sat_idx=ii, tar_idx=jj, dlo_idx=rrp
                    )
                    qubo_key = (row_idx, col_idx)
                    dlo_dlo_couplings[qubo_key] = (
                        dlo_dlo_couplings.get(qubo_key, 0) + self.lambda6
                    )

        if self._mp4eo_instance.get_preprocessed_constraint_flag("C7"):
            for ii in range(self.n_satellites):
                for jj in range(self.n_targets):
                    qij = self._mp4eo_instance.qij(ii, jj)
                    Lij = self._mp4eo_instance.get_preprocessed_c7_idxs(ii, jj)

                    for jjp in range(self.n_targets):
                        qijp = self._mp4eo_instance.qij(ii, jjp)
                        Lijp = self._mp4eo_instance.get_preprocessed_c7_idxs(
                            ii, jjp
                        )
                        cte_7qq = self.lambda7 * qij * qijp

                        for _, rr in Lij:
                            row_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                                sat_idx=ii, tar_idx=jj, dlo_idx=rr
                            )

                            for _, rrp in Lijp:
                                col_idx = (
                                    self._mp4eo_instance._dlo_idxs_to_sg_idx(
                                        sat_idx=ii, tar_idx=jjp, dlo_idx=rrp
                                    )
                                )

                                qubo_key = (row_idx, col_idx)
                                dlo_dlo_couplings[qubo_key] = (
                                    dlo_dlo_couplings.get(qubo_key, 0)
                                    + cte_7qq
                                )

        # Output
        return dlo_dlo_couplings

    def _compute_dto_dlo_qubo_couplings(self):
        """Computes and returns the DTO-DLO binary-interaction couplings."""

        dto_dlo_couplings = {}
        sigma = self._mp4eo_instance.sigma

        for ii in range(self.n_satellites):
            si = sigma[ii]
            for jj in range(self.n_targets):
                thij = self._mp4eo_instance.thetaij(ii, jj)
                for kk in range(thij):
                    row_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                        sat_idx=ii, tar_idx=jj, dto_idx=kk
                    )
                    for rr in range(si):
                        col_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dlo_idx=rr
                        )
                        qubo_key = (row_idx, col_idx)
                        dto_dlo_couplings[qubo_key] = (
                            dto_dlo_couplings.get(qubo_key, 0)
                            - 2 * self.lambda3
                        )

                if self._mp4eo_instance.get_preprocessed_constraint_flag("C4"):
                    Fij = self._mp4eo_instance.get_preprocessed_c4_idxs(ii, jj)
                    for kk, rr in Fij:
                        row_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                            ii, jj, kk
                        )
                        col_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                            ii, jj, rr
                        )
                        qubo_key = (row_idx, col_idx)
                        dto_dlo_couplings[qubo_key] = (
                            dto_dlo_couplings.get(qubo_key, 0) + self.lambda4
                        )

        if self._mp4eo_instance.get_preprocessed_constraint_flag("C7"):
            for ii in range(self.n_satellites):
                for jj in range(self.n_targets):
                    qij = self._mp4eo_instance.qij(ii, jj)
                    Lij = self._mp4eo_instance.get_preprocessed_c7_idxs(ii, jj)

                    for jjp in range(self.n_targets):
                        qijp = self._mp4eo_instance.qij(ii, jjp)
                        Lijp = self._mp4eo_instance.get_preprocessed_c7_idxs(
                            sat_idx=ii, tar_idx=jjp
                        )
                        cte_7qq = self.lambda7 * qij * qijp

                        for kk, rr in Lij:
                            for kkp, rrp in Lijp:
                                row_idx = (
                                    self._mp4eo_instance._dto_idxs_to_sg_idx(
                                        sat_idx=ii, tar_idx=jj, dto_idx=kk
                                    )
                                )
                                col_idx = (
                                    self._mp4eo_instance._dlo_idxs_to_sg_idx(
                                        sat_idx=ii, tar_idx=jjp, dlo_idx=rrp
                                    )
                                )
                                qubo_key = (row_idx, col_idx)
                                dto_dlo_couplings[qubo_key] = (
                                    dto_dlo_couplings.get(qubo_key, 0)
                                    - cte_7qq
                                )

                                row_idx = (
                                    self._mp4eo_instance._dto_idxs_to_sg_idx(
                                        sat_idx=ii, tar_idx=jjp, dto_idx=kkp
                                    )
                                )
                                col_idx = (
                                    self._mp4eo_instance._dlo_idxs_to_sg_idx(
                                        sat_idx=ii, tar_idx=jj, dlo_idx=rr
                                    )
                                )
                                qubo_key = (row_idx, col_idx)
                                dto_dlo_couplings[qubo_key] = (
                                    dto_dlo_couplings.get(qubo_key, 0)
                                    - cte_7qq
                                )

        # Output
        return dto_dlo_couplings

    def _compute_dto_slack_qubo_couplings(self):
        """Computes and returns the DTO-slack binary-interaction couplings."""

        dto_slack_couplings = {}

        for ii in range(self.n_satellites):
            nsi = self._mp4eo_instance.nslacki(ii)

            for jj in range(self.n_targets):
                cte_7q = 2 * self.lambda7 * self._mp4eo_instance.qij(ii, jj)
                Lij = self._mp4eo_instance.get_preprocessed_c7_idxs(ii, jj)

                for aa in range(nsi):
                    col_idx = self._mp4eo_instance._slack_idxs_to_sg_site(
                        sat_idx=ii, slack_idx=aa
                    )
                    cstrength = 2**aa * cte_7q

                    for kk, _ in Lij:
                        row_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dto_idx=kk
                        )
                        qubo_key = (row_idx, col_idx)
                        dto_slack_couplings[qubo_key] = (
                            dto_slack_couplings.get(qubo_key, 0) + cstrength
                        )

        # Output
        return dto_slack_couplings

    def _compute_dlo_slack_qubo_couplings(self):
        """Computes and returns the DLO-slack binary-interaction couplings."""

        dlo_slack_couplings = {}

        for ii in range(self.n_satellites):
            nsi = self._mp4eo_instance.nslacki(ii)

            for jj in range(self.n_targets):
                cte_7q = 2 * self.lambda7 * self._mp4eo_instance.qij(ii, jj)
                Lij = self._mp4eo_instance.get_preprocessed_c7_idxs(ii, jj)

                for aa in range(nsi):
                    col_idx = self._mp4eo_instance._slack_idxs_to_sg_site(
                        sat_idx=ii, slack_idx=aa
                    )
                    cstrength = 2**aa * cte_7q

                    for _, rr in Lij:
                        row_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                            sat_idx=ii, tar_idx=jj, dlo_idx=rr
                        )
                        qubo_key = (row_idx, col_idx)
                        dlo_slack_couplings[qubo_key] = (
                            dlo_slack_couplings.get(qubo_key, 0) - cstrength
                        )

        # Output
        return dlo_slack_couplings

    def _compute_slack_slack_qubo_couplings(self):
        """
        Computes and returns the slack-slack binary-interaction
        couplings.
        """

        slack_slack_couplings = {}
        cte_7 = 2 * self.lambda7

        for ii in range(self.n_satellites):
            nsi = self._mp4eo_instance.nslacki(ii)

            for aa in range(nsi):
                cstrength = 2**aa * cte_7
                row_idx = self._mp4eo_instance._slack_idxs_to_sg_site(
                    sat_idx=ii, slack_idx=aa
                )

                for aap in range(aa + 1, nsi):
                    col_idx = self._mp4eo_instance._slack_idxs_to_sg_site(
                        satellite_idx=ii, slack_idx=aap
                    )
                    slack_slack_couplings[(row_idx, col_idx)] = (
                        2**aap * cstrength
                    )

        # Output
        return slack_slack_couplings

    def construct_qubo_matrix(self, rescale_primal_couplings=True):
        """
        Constructs the QUBO matrix associated to
        :attr:`~QUBOMapper._mp4eo_instance`.

        Arguments
        ---------
        rescale_primal_couplings : bool, optional
            Whether to rescale the primal couplings that
            define the cost function of the primal optimization
            problem to compute QUBO matrix elements within a
            more numerically stable framework.
            Default to :obj:`True`, i.e., the primal couplings
            will be rescaled.

        Returns
        -------
        :obj:`None`

        Details
        -------
        The QUBO problem is completely described by the offset
        :math:`Cte`, which is not required to find the minimizer
        of the QUBO cost function, and a square symmetric matrix
        :math:`\mathcal{Q}`. Given its symmetry, we compute only
        the upper triangular part of :math:`\mathcal{Q}`.
        """

        nq = self.get_mspace_dim
        qubo_matrix = np.zeros((nq, nq))

        # Compute the diagonal terms, i.e., the binary biases
        for idx, mel in self._compute_dto_qubo_biases(
            rescale_primal_couplings
        ).items():
            qubo_matrix[idx][idx] += mel
        for idx, mel in self._compute_dlo_qubo_biases(
            rescale_primal_couplings
        ).items():
            qubo_matrix[idx][idx] += mel

        # Compute off-diagonal matrix elements,
        # i.e., the binary-binary interaction couplings
        for idxs, mel in self._compute_dto_dto_qubo_couplings().items():
            qubo_matrix[idxs[0]][idxs[1]] += mel
        for idxs, mel in self._compute_dlo_dlo_qubo_couplings().items():
            qubo_matrix[idxs[0]][idxs[1]] += mel
        for idxs, mel in self._compute_dto_dlo_qubo_couplings().items():
            qubo_matrix[idxs[0]][idxs[1]] += mel

        # Handle auxiliary slack variables
        if self._mp4eo_instance.get_preprocessed_constraint_flag("C7"):
            ## Compute the diagonal terms, i.e., the binary biases
            for idx, mel in self._compute_slack_qubo_biases().items():
                qubo_matrix[idx][idx] += mel

            ## Compute off-diagonal matrix elements,
            ## i.e., the binary-binary interaction couplings
            ## DTO-slack interaction couplings
            for idxs, mel in self._compute_dto_slack_qubo_couplings().items():
                qubo_matrix[idxs[0]][idxs[1]] += mel

            ## DLO-slack interaction couplings
            for idxs, mel in self._compute_dlo_slack_qubo_couplings().items():
                qubo_matrix[idxs[0]][idxs[1]] += mel

            ## slack-slack interaction couplings
            for (
                idxs,
                mel,
            ) in self._compute_slack_slack_qubo_couplings().items():
                qubo_matrix[idxs[0]][idxs[1]] += mel

        # Symmetrize and update the class instance attribute
        qubo_upper = np.triu(qubo_matrix)
        qubo_lower = np.tril(qubo_matrix, -1).T
        self._qubo_matrix = qubo_upper + qubo_lower

    def to_spinglass_couplings(self):
        r"""
        Derives the corresponding spin glass model from the QUBO
        problem reformulation.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        The couplings of the spin glass Hamiltonian are obtained from
        the QUBO matrix elements by transforming binary variables to
        spin-:math:`1 \mathbin{/} 2` variables via the following linear
        transformation:

        .. math::

            x = \dfrac{1 + \sigma}{2}

        where :math:`x \in \{0, 1\}` is the binary variable and
        :math:`\sigma \in \{-1, 1\}` is the classical spin variable.

        The transformation implementing the mapping between the QUBO
        cost function and the spin glass Hamiltonian is as follows:

        .. math::

            \begin{align}
            & \mathcal{E}_0 = Cte
            + \dfrac{1}{2} \sum_{j=1}^{n_Q} \mathcal{Q}_{jj}
            + \dfrac{1}{2} \sum{k > j}^{n_Q} \mathcal{Q}_{jk} \\
            & \mathrm{h}_j = \dfrac{1}{2} \mathcal{Q}_{jj}
            + \dfrac{1}{2} \sum_{k \neq j, k=1}^{n_Q} \mathcal{Q}_{jk} \\
            & \mathrm{J}_{jk} = \dfrac{1}{2} \mathcal{Q}_{jk}
            \end{align}

        and the spin glass Hamiltonian reads as

        .. math::

            \mathcal{H}
            =
            \mathcal{E}_0
            + \sum_{j=1}^{n_Q} \mathrm{h}_j \sigma_{j}
            + \sum_{j < k}^{n_Q} \mathrm{J}_{jk} \sigma_j \sigma_k

        Specifically, the spin glass coupling defining the model are
        arranged as the following python dictionary:
        - ``offset``: the constant term :math:`\mathcal{E}_0` proportional
                    to the identity, which is just an energy shift.
        - ``one-body``: the set of single-body couplings :math:`\mathrm{h}_j`,
                        i.e., the set of local longitudinal magnetic fields
                        (biases), one for each spin-:math:`1 \mathbin{/} 2` in
                        the spin glass system;
        - ``two-body``: the set of two-body (in general all-to-all) couplings
                        :math:`\mathrm{J}_{jk}` describing the interactions
                        between pairs of spin-:math:`1 \mathbin{/} 2`.
        """

        qjj = np.diagonal(self.qubo_matrix).copy()
        qij_upper = np.triu(self.qubo_matrix, k=1)

        # Energy offset
        cte = self.offset + 0.25 * np.sum(qjj) + 0.25 * np.sum(qij_upper)

        # One-body couplings, aka, spin glass local magnetic fields
        magnetic_fields = 0.5 * qjj
        for jj in range(self.get_mspace_dim):
            magnetic_fields[jj] += 0.25 * np.sum(qij_upper[jj, jj + 1 :])
            magnetic_fields[jj] += 0.25 * np.sum(qij_upper[:jj, jj])

        # Update the corresponding class instance attribute
        self._spinglass_couplings = {
            "offset": cte,
            "one-body": magnetic_fields,
            "two-body": 0.25 * qij_upper,
        }

    def _set_qubo_penalty_constants(self, **kwargs):
        r"""
        Computes penalty constants :math:`\lambda_1, \cdots, \lambda_7`
        to be used in the MP4EO QUBO reformulation. These constants are
        necessary to enforce optimization constraints.

        Arguments
        ---------
        **kwargs : Dict[str, Any]
            Additional optional arguments, e.g., to change values
            to the QUBO penalty constants before constructing the
            QUBO matrix.

            To tune the :math:`\lambda`s based on a heuristic related
            to the couplings of the cost function for the primal
            optimization problem, use the multiplicative penalty ratios
            as optional arguments:

            - ``lambda1_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_1` related to constraint (1).
                Default to :obj:`None`.

            - ``lambda2_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_2` related to constraint (2).
                Default to :obj:`None`.

            - ``lambda3_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_3` related to constraint (3).
                Default to :obj:`None`.

            - ``lambda4_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_4` related to constraint (4).
                Default to :obj:`None`.

            - ``lambda5_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_5` related to constraint (5).
                Default to :obj:`None`.

            - ``lambda6_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_6` related to constraint (6).
                Default to :obj:`None`.

            - ``lambda7_ratio`` : float
                New value for the ratio of the QUBO penalty constant
                :math:`\lambda_7` related to constraint (7).
                Default to :obj:`None`.

            In this case, penalty constants are defined as

            .. math::

            \lambda_{\ell}
            =
            \bar{\lambda}_{\ell}
            \mathcal{C}\left(\boldsymbol{\omega}\right)

            where :math:`\bar{\lambda}_{\ell} \in \mathbb{R}` is the ratio
            that can be tuned with the above “ratio” arguments, while
            :math:`\mathcal{C}\left(\boldsymbol{\omega}\right) \in \mathbb{R}`
            is some average related to the primal problem couplings
            :math:`\boldsymbol{\omega}` that defines the cost function.
            The way this last quantity is computed can be set via the additional
            optional arguments:

            - ``baseline``: str
                Use `mean` to set :math:`\mathcal{C}` as the mean value of
                the primal cost function couplings;
                `max` to use as baseline for the tuning of the QUBO penalty
                constants the maximum value of the couplings.
                The default is `mean`.

            Alternatively, to directly set the penalty constant values, use the
            following optional arguments:

            - ``lambda1`` : float
                New value for the QUBO penalty constant :math:`\lambda_1`
                related to constraint (1).
                Default to :obj:`None`.

            - ``lambda2`` : float
                New value for the QUBO penalty constant :math:`\lambda_2`
                related to constraint (2).
                Default to :obj:`None`.

            - ``lambda3`` : float
                New value for the QUBO penalty constant :math:`\lambda_3`
                related to constraint (3).
                Default to :obj:`None`.

            - ``lambda4`` : float
                New value for the QUBO penalty constant :math:`\lambda_4`
                related to constraint (4).
                Default to :obj:`None`.

            - ``lambda5`` : float
                New value for the QUBO penalty constant :math:`\lambda_5`
                related to constraint (5).
                Default to :obj:`None`.

            - ``lambda6`` : float
                New value for the QUBO penalty constant :math:`\lambda_6`
                related to constraint (6).
                Default to :obj:`None`.

            - ``lambda7`` : float
                New value for the QUBO penalty constant :math:`\lambda_7`
                related to constraint (7).
                Default to :obj:`None`.

            If you need to rescale the couplings of the primal optimization
            problem cost function for numerical stability, you can provide
            an optional boolen argument ``rescale_primal_couplings``, whose
            default is set to :obj:`True`.

        Returns
        -------
        :obj:`None`

        Raises
        ------
        :class:`mp4eo_optimization.exceptions.MP4EOQUBOMapperError`
            If any optional arguments is set to an invalid value.
        """

        # Read optional arguments to choose the strategy
        ratios = np.array(
            [kwargs.get(f"lambda{ii}_ratio") for ii in range(1, 8)]
        )
        lambdas = np.array([kwargs.get(f"lambda{ii}") for ii in range(1, 8)])
        strategy = kwargs.get("baseline", "mean")
        rescale_flag = kwargs.get("rescale_primal_couplings", True)

        # Select how to tune the penalty constants
        if np.all(ratios == None) and np.all(lambdas == None):
            penalty_oom = self._compute_penalty_constant_baseline(
                strategy=strategy, rescale_primal_couplings=rescale_flag
            )
            self._penalties_cte = {
                cname: clambda[1] * penalty_oom if clambda[0] else 0
                for cname, clambda in self._mp4eo_instance.constraints.items()
            }
            return

        if np.all(ratios == None):
            for ii, new_lambda in enumerate(lambdas, start=1):
                if new_lambda is not None:
                    setattr(self, f"lambda{ii}", new_lambda)
            return

        penalty_oom = self._compute_penalty_constant_baseline(
            strategy=strategy, rescale_primal_couplings=rescale_flag
        )
        for ii, ratio in enumerate(ratios, start=1):
            if ratio is not None:
                self._mp4eo_instance.update_constraint(f"C{ii}", ratio)
        self._penalties_cte = {
            cname: clambda[1] * penalty_oom if clambda[0] else 0
            for cname, clambda in self._mp4eo_instance.constraints.items()
        }

    def _compute_penalty_constant_baseline(
        self, strategy="mean", rescale_primal_couplings=True
    ):
        """
        Computes the baseline value to tune the QUBO penalty
        constants using ratios.

        Arguments
        ---------
        strategy : str, optional
            How to compute the baseline value.
            Use ``mean`` to set the baseline to the average
            value of the primal cost function couplings; use
            ``max`` to use the maximum value of the couplings.
            Default to ``mean``.

        rescale_primal_couplings : bool, optional
            Whether to rescale the primal couplings that
            define the cost function of the primal optimization
            problem to compute QUBO matrix elements within a
            more numerically stable framework.
            Default to :obj:`True`, i.e., the primal couplings
            will be rescaled.

        Returns
        -------
        float
            The baseline value to define an order of magnitude
            for the penalty constant values.

        Raises
        ------
        :class:`mp4eo_optimization.exceptions.MP4EOQUBOMapperError`
            If ``strategy`` is set to an invalid value.
        """

        w_dto = self._mp4eo_instance.omega_dto.copy()
        w_dlo = self._mp4eo_instance.omega_dlo.copy()
        if rescale_primal_couplings:
            factor = self._primal_couplings_rescaling_factor
            w_dto /= factor
            w_dlo /= factor

        if strategy == "mean":
            return (np.mean(w_dto) + np.mean(w_dlo)) * 0.5
        if strategy == "max":
            return max(np.max(w_dto), np.max(w_dlo))

        raise MP4EOQUBOMapperError(
            f"Invalid strategy '{strategy}' to tune the QUBO penalty constants."
        )

    def save_qubo_matrix(self, filename="qubo_matrix", extension="json"):
        """
        Saves the QUBO matrix in various formats with metadata.

        Arguments
        ---------
        filename : str, optional
            Name of the output log file for storing the QUBO matrix.
            Default to ``qubo_matrix``.
            The file will be created in :attr:`~QUBOMapper.log_dir`.

        extension : str, optional
            Extension of the serialized QUBO matrix file.
            Default to ``json``.

        Returns
        -------
        :obj:`None`

        Details
        -------
        Available file extensions:

        - ``pkl``: serializes the QUBO matrix through :mod:`pickle`;
        - ``npy``: uses :mod:`numpy` native binary format;
        - ``txt``: human-readable text matrix, inefficient for large matrices;
        - ``json``: portable serialization through :mod:`json` (suggested);

        In the ``txt`` case, we store only the upper triangular part of the
        QUBO matrix, but keep in mind that the symmetrized factor of 2 is
        already incorporated in the matrix element.
        """

        qubo_mat = self.qubo_matrix
        available_extensions = {"json", "pkl", "npz", "txt"}
        extension = extension.lower()

        # Check the rerquired extension
        if extension not in available_extensions:
            raise ValueError(
                f"Unsupported file extension '{extension}' for serializing",
                "the QUBO matrix.",
            )

        # Handle and check the desidered name for the log file
        if any(filename.endswith(f".{ext}") for ext in available_extensions):
            filepath = self.log_dir / filename
            extension = filepath.suffix[1:]
            filename = filepath.stem
        else:
            filepath = self.log_dir / f"{filename}.{extension}"

        # Prepare metadata for reading a-posteriori
        qubo_nvars = self.get_mspace_dim
        qubo_naux = self._mp4eo_instance.n_slack_binaries
        qubo_shift = self.offset
        primal_resc = self._primal_couplings_rescaling_factor
        metadata = {
            "qubo_matrix_size": (qubo_nvars, qubo_nvars),
            "number_of_decision_variables": qubo_nvars,
            "number_of_auxiliary_variables": qubo_naux,
            "offset": qubo_shift,
            "primal_rescaling": primal_resc,
        }

        # JSON
        if extension == "json":
            qubo_list = qubo_mat.tolist()
            log_data = {"metadata": metadata, "qubo_matrix": qubo_list}
            with open(filepath, "w") as fh:
                json.dump(log_data, fh, indent=4, separators=(", ", ": "))

        # PICKLE
        elif extension == "pkl":
            log_data = {"metadata": metadata, "qubo_matrix": qubo_mat}
            with open(filepath, "wb") as fh:
                pickle.dump(log_data, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # NUMPY
        elif extension == "npz":
            np.savez(filepath, qubo_matrix=qubo_mat, metadata=metadata)

        # HUMAN-READABLE
        elif extension == "txt":
            with open(filepath, "w") as fh:
                fh.write("## Row. 1: QUBO matrix size\n")
                fh.write("## Row. 2: Number of decision variables\n")
                fh.write("## Row. 3: Number of auxiliary variables\n")
                fh.write("## Row. 4: QUBO energy offset\n")
                fh.write(
                    "## Row. 5: Rescaling factor of the primal couplings\n"
                )
                fh.write("## Row. 6 -->\n")
                fh.write("##   Col. 1: QUBO matrix row index j\n")
                fh.write("##   Col. 2: QUBO matrix column index k (k >= j)\n")
                fh.write("##   Col. 3: QUBO matrix element Q_jk\n")
                fh.write("#" * 79 + "\n\n")
                fh.write(f"({qubo_nvars}, {qubo_nvars})\n{qubo_nvars}\n")
                fh.write(f"{qubo_naux}\n{qubo_shift:.6f}\n{primal_resc:.6f}\n")
                for (row_idx, col_idx), Qij in np.ndenumerate(qubo_mat):
                    if col_idx >= row_idx:
                        fh.write(f"{row_idx}\t{col_idx}\t{Qij:.6f}\n")

    def save_spinglass_model(
        self, filename="spinglass_model", extension="json"
    ):
        """
        Saves the spin glass couplings in various formats with metadata.

        Arguments
        ---------
        filename : str, optional
            Name of the output log file for storing the spin glass model.
            Default to ``spinglass_model``.
            The file will be created in :attr:`~QUBOMapper.log_dir`.

        extension : str, optional
            Extension of the serialized spin glass model file.
            Default to ``json``.

        Returns
        -------
        :obj:`None`

        Details
        -------
        Available file extensions:

        - ``pkl``: serializes the QUBO matrix through :mod:`pickle`;
        - ``npy``: uses :mod:`numpy` native binary format;
        - ``txt``: human-readable text matrix;
        - ``json``: portable serialization through :mod:`json` (suggested);

        In the ``txt`` case, we store only the non-zero couplings needed
        to construct the full spin-glass Hamiltonian.
        """

        sg_hfields = self.spinglass_hfields
        sg_interactions = self.spinglass_interactions
        available_extensions = {"json", "pkl", "npz", "txt"}
        extension = extension.lower()

        # Check the rerquired extension
        if extension not in available_extensions:
            raise ValueError(
                f"Unsupported file extension '{extension}' for serializing",
                "the spin glass model.",
            )

        # Handle and check the desidered name for the log file
        if any(filename.endswith(f".{ext}") for ext in available_extensions):
            filepath = self.log_dir / filename
            extension = filepath.suffix[1:]
            filename = filepath.stem
        else:
            filepath = self.log_dir / f"{filename}.{extension}"

        # Prepare metadata for reading a-posteriori
        nspins = self.get_mspace_dim
        nslacks = self._mp4eo_instance.n_slack_binaries
        sg_shift = self.spinglass_shift
        metadata = {
            "number_of_spins": nspins,
            "number_of_auxiliary_spins": nslacks,
            "offset": sg_shift,
        }

        # JSON
        if extension == "json":
            log_data = {
                "metadata": metadata,
                "one-body": sg_hfields.tolist(),
                "two-body": sg_interactions.tolist(),
            }
            with open(filepath, "w") as fh:
                json.dump(log_data, fh, indent=4, separators=(", ", ": "))

        # PICKLE
        elif extension == "pkl":
            log_data = {
                "metadata": metadata,
                "one-body": sg_hfields,
                "two-body": sg_interactions,
            }
            with open(filepath, "wb") as fh:
                pickle.dump(log_data, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # NUMPY
        elif extension == "npz":
            np.savez(
                filepath,
                one_body=sg_hfields,
                two_body=sg_interactions,
                metadata=metadata,
            )

        # HUMAN-READABLE
        elif extension == "txt":
            with open(filepath, "w") as fh:
                fh.write("## Row. 1: Number of spins\n")
                fh.write("## Row. 2: Number of decision variables\n")
                fh.write("## Row. 3: Number of auxiliary spins\n")
                fh.write("## Row. 4: Spin glass energy offset\n")
                fh.write("## Row. 5 -->\n")
                fh.write("##   Col. 1: spin-j index\n")
                fh.write("##   Col. 2: spin-k index (k >= j)\n")
                fh.write("##   Col. 3: spin glass coupling strength\n")
                fh.write("#" * 79 + "\n\n")
                fh.write(f"{nspins}\n{nspins}\n{nslacks}\n{sg_shift:.6f}\n")
                for jj, hh in enumerate(sg_hfields):
                    fh.write(f"{jj}\t{jj}\t{hh:.6f}\n")
                for (jj, kk), Jjk in np.ndenumerate(sg_interactions):
                    if kk >= jj:
                        fh.write(f"{jj}\t{kk}\t{Jjk:.6f}\n")


class QuditsMapper(MP4EOMapper):
    r"""
    Mapping an MP4EO instance to a system of qudits using a **single-excitation**
    local basis encoding.

    Parameters
    ----------
    mp4eo_instance : mp4eo_optimization.mp4eo_instance.MP4EOInstance
        MP4EO problem instance to map into a many-qudit system.

    Attributes
    ----------
    Same as the base class :class:`~MP4EOMapper`.

    Attributes: single-excitation basis encoding
    --------------------------------------------
    _dto_local_dimensions : List[int]
        Local dimensions of the qudits representing DTO
        binaries in the original MP4EO model.

    _dlo_local_dimensions : List[int]
        Local dimensions of the qudits representing DLO
        binaries in the original MP4EO model.

    _dto_local_basis : List[np.ndarray[str]]
        Set of single-excitation local basis states for each
        DTO qudit in the mapped MP4EO model.

    _dlo_local_basis : List[np.ndarray[str]]
        Set of single-excitation local basis states for each
        DLO qudit in the mapped MP4EO model.

    Attributes: Hamiltonian encoding
    --------------------------------
    _dto_local_single_body_projectors : Dict[int, scipy.sparse.dia_matrix]
        Set of local projectors on single DTO-qudit logical states.
        Used to encode the original mission planning problem into a
        many-qudit Hamiltonian.
        Keys are DTO site indices, values are the corresponding projectors
        acting on each single site, derived from both cost function and
        constraints contributions.

    _dlo_local_single_body_projectors : Dict[int, scipy.sparse.dia_matrix]
        Set of local projectors on single DLO-qudit logical states.
        Used to encode the original mission planning problem into a
        many-qudit Hamiltonian.
        Keys are DLO site indices, values are the corresponding projectors
        acting on each single site, derived from both cost function and
        constraints contributions.

    _dto_local_two_body_projectors : Dict[Tuple[int, int], scipy.sparse.dia_matrix]
        Set of local projectors on two interacting DTO-qudit Hilbert
        spaces.
        Used to encode non-local constraints in the original mission
        planning problem as a many-qudit interaction Hamiltonian.
        Keys are pairs of DTO site indices, values are the corresponding
        two-body projectors acting on the two Hilbert spaces.

    _dlo_local_two_body_projectors : Dict[Tuple[int, int], scipy.sparse.dia_matrix]
        Set of local projectors on two interacting DLO-qudit Hilbert
        spaces.
        Used to encode non-local constraints in the original mission
        planning problem as a many-qudit interaction Hamiltonian.
        Keys are pairs of DLO site indices, values are the corresponding
        two-body projectors acting on the two Hilbert spaces.

    _dto_dlo_local_two_body_projectors : Dict[Tuple[int, int], scipy.sparse.dia_matrix]
        Set of local projectors on two interacting DTO-DLO-qudit Hilbert
        spaces.
        Used to encode non-local constraints in the original mission
        planning problem as a many-qudit interaction Hamiltonian.
        Keys are pairs of DTO and DLO site indices, values are the
        corresponding two-body projectors acting on the two Hilbert spaces.

    Attributes: observables
    -----------------------
    _dto_qudit_local_observables : Dict[str, List[scipy.sparse.dia_matrix]]
        Set of local DTO-qudit observables to measure.
        Keys are names of the observables, values are lists of operators
        representing the observable to be measure on each DTO-qudit site.

    _dlo_qudit_local_observables : Dict[str, List[scipy.sparse.dia_matrix]]
        Set of local DTL-qudit observables to measure.
        Keys are names of the observables, values are lists of operators
        representing the observable to be measure on each DLO-qudit site.

    Details
    -------
    For each target indexed by :math:`j = 0, 1, \cdots, M - 1`,
    we construct a **DTO** and a **DLO qudit** such that mission
    constraints (1) and (2) are automatically satisfied by the
    specific local basis describing each physical qudit site.
    Specifically, the local basis of states for the :math:`j`-th
    qudit is defined as a **single-excitation** basis by grouping
    all the DTO binary configuration related to the :math:`j`-th
    target as follows:

    .. math::

        {\vert \mathcal{X} \rangle}_j
        =
        \vert
        x_{0j0} x_{0j1} \cdots x_{0j, \vartheta_{0j} - 1}
        x_{1j0} x_{1j1} \cdots x{1j, \vartheta_{1j}-1}
        \cdots x_{Nj, \vartheta_{Nj} - 1}
        \rangle

    We then discard binary configurations that violate constraint
    (1), i.e., bit-strings containing more than one ``1``.
    The remaining valid bit-strings define the :math:`D_j` logical
    single-excitation basis states of the :math:`j`-th DTO qudit.
    These states are denoted as :math`{\vert 0_x \rangle}_j`:,
    :math`{\vert 1_x \rangle}_j`:, :math:`\cdots {\vert D^{(j)}_x - 1 \rangle}_j`,
    where :math:`D_j` represents the local dimension of the Hilbert
    space for the :math:`j`-th DTO qudit.

    The same procedure is applied to the DLO qudits, where the local
    basis is defined as:

    .. math::

        {\vert \mathcal{Y} \rangle}_j
        =
        \vert
        y_{0j0} y_{0j1} \cdots y_{0j, \vartheta_{0j} - 1}
        y_{1j0} y_{1j1} \cdots y_{1j, \vartheta_{1j} - 1}
        \cdots y_{Nj, \vartheta_{Nj} - 1}
        \rangle

    Thus, for each target :math:`j`, there are two physical sites:
    one for the DTO qudit and one for the DLO qudit.
    The state associated with the **mission site** :math:`j` is
    defined as the product state of the two qudits

    .. math::

        {\vert \mathcal{X} \mathcal{Y} \rangle}_j
        =
        {\vert \mathcal{X} \rangle}_j
        \otimes
        {\vert \mathcal{Y} \rangle}_j

    The resulting mapped many-body system is therefore defined by
    :math:`2M` qudits. Additional mission constraints can be
    implemented as qudit-qudit interactions, expressed as projectors
    on the local basis states of the qudits.

    Notes
    -----
    The qudit-based encoding is a **single-excitation** local
    basis for each qudit. Indeed, in our current implementation,
    for each qudit (representing all binary variables for a fixed
    target index j, the local basis consists of bitstrings with a
    **single 1** and the rest 0.
    That is, for a qudit of local dimension :math:`D`, each logical
    state represents a single-excitation because only one of the
    possible “*excitations*” (binary 1s) is turned on at a time.

    This mapper provides only the definition of the local
    basis and local operators for each qudit degree of freedom
    to encode bunch of MP4EO binaries as multi-level system.
    Corresponding many-qudit operators and Hamiltonian are
    implemented in the :mod:`mp4eo_optimization.mp4eo_solvers`
    submodule depending on the selected solver.
    """

    # ---------------------------------------------------------------------- #
    #                              Constructor                               #
    # ---------------------------------------------------------------------- #
    penalty = None
    def __init__(self, mp4eo_instance, build_mode="both"):
        """Initializes the quantum-ispired qudits-based mapper."""

        # Inherit base-class constructor
        super().__init__(mp4eo_instance, "Qudit Mapper")
        self._build_exact = (build_mode in ("both", "exact"))
        self._build_tn    = (build_mode in ("both", "tn"))

        # Compute local dimensions
        self._dto_local_dimensions = [  # DTO sites
            sum(
                self._mp4eo_instance.thetaij(sat_idx, tar_idx)
                for sat_idx in range(self.n_satellites)
            )
            for tar_idx in range(self.n_targets)
        ]

        dlo_local_dim = sum(  # DLO sites
            self._mp4eo_instance.sigma[sat_idx]
            for sat_idx in range(self.n_satellites)
        )
        self._dlo_local_dimensions = [dlo_local_dim] * self.n_targets

        # Construct single-excitation logical basis
        self._dto_local_basis = [  # Truncate basis states violating constraints (1)
            np.array([f"{'0' * dd}1{'0' * (Dj - dd - 1)}" for dd in range(Dj)])
            for Dj in self._dto_local_dimensions
        ]

        logical_state = np.array(  # Truncate basis states violating constraints (2)
            [f"{'0' * dd}1{'0' * (dlo_local_dim - dd - 1)}" for dd in range(dlo_local_dim)]
        )
        self._dlo_local_basis = [logical_state] * self.n_targets

        # self.build_all_projectors(penalty)
    def set_build_mode(self, build_mode: str):
        bm = (build_mode or "both").lower()
        self._build_exact = (bm in ("both", "exact"))
        self._build_tn    = (bm in ("both", "tn"))

    def build_all_projectors(self, penalty=None):
        # self._build_exact = bool(for_exact)
        # self._build_tn    = bool(for_tn)

        self._construct_dto_cost_func_projectors()
        self._construct_dlo_cost_func_projectors()
        self._construct_number_operators()

        self._construct_C3_projectors(penalty=penalty)
        self._construct_C4_projectors(penalty=penalty)
        self._construct_C5_projectors(penalty=penalty)
        self._construct_C6_projectors(penalty=penalty)
    # ---------------------------------------------------------------------- #
    #                               Properties                               #
    # ---------------------------------------------------------------------- #
    @property
    def get_mspace_dim(self):
        """
        Returns the number of MP4EO qudits in the mapped
        quantum many-body system.
        """
        return 2 * self.n_targets

    @property
    def hilbert_space_dim(self):
        """Returns the dimension of the Hilbert space after the mapping."""
        dim = math.prod(self._dto_local_dimensions)
        dim *= math.prod(self._dlo_local_dimensions)
        return dim

    @property
    def local_dimensions(self):
        """
        Returns local dimensions of the whole many-qudit system
        after joining DTO and DLO sites one after the other.
        """
        return [
            dim for pair in zip(self._dto_local_dimensions, self._dlo_local_dimensions)
            for dim in pair
        ]
    @property
    def dto_local_dimensions(self):
        """
        Returns local dimensions of the whole many-qudit system
        after joining DTO and DLO sites one after the other.
        """
        return self._dto_local_dimensions
    
    @property
    def dlo_local_dimensions(self):
        """
        Returns local dimensions of the whole many-qudit system
        after joining DTO and DLO sites one after the other.
        """
        return self._dlo_local_dimensions
        
    @property
    def dto_local_basis(self):
        """Read-only access to all the DTO qudit local basis."""
        return self._dto_local_basis

    @property
    def dlo_local_basis(self):
        """Read-only access to all the DLO qudit local basis."""
        return self._dlo_local_basis

    @property
    @ensure_init("_C3_projectors")
    def C3_projectors(self):
        """Read-only access to the constraint 3 (C3) projectors."""
        return self._C3_projectors

    @property
    @ensure_init("_C4_projectors")
    def C4_projectors(self):
        """Read-only access to the constraint 4 (C4) projectors."""
        return self._C4_projectors

    @property
    @ensure_init("_C5_projectors")
    def C5_projectors(self):
        """Read-only access to the constraint 5 (C5) projectors."""
        return self._C5_projectors

    @property
    @ensure_init("_C6_projectors")
    def C6_projectors(self):
        """Read-only access to the constraint 6 (C6) projectors."""
        return self._C6_projectors

    # ---------------------------------------------------------------------- #
    #                             Static Methods                             #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _get_single_qudit_projectors(local_dimension, selected_level = None):
        r"""
        Returns the list of the projectors (onto the logical basis) for a
        single qudit.

        Arguments
        ---------
        local_dimension : int
            Local dimension of the Hilbert space of the qudit.

        selected_levels : Optional[Iterable[int]]
            Indices of levels for which to generate projectors. If None,
            all levels [0, D-1] are used.

        Returns
        -------
        List[scipy.sparse.dia_matrix]
            List of logical-state projectors.

        Details
        -------
        For a single :math:`\mathrm{D}`-level qudit with local logical basis
        states

        .. math::

            \left\{
            \lvert{\ell}\rangle \mid \ell = 0, 1, \cdots, \mathrm{D} - 1
            \right\}

        we define the local logical-state projector as the **diagonal**
        operator:

        .. math::

            \Pi_{\ell} = \lvert{\ell}\rangle \langle{\ell}\rvert

        which is a :math:`\mathrm{D} \times \mathrm{D}` matrix with a
        single 1 at position :math:`\left(\ell, \ell\right)`, and zeros
        elsewhere.

        Each qudit has, of course, :math:`\mathrm{D} =` ``local_dimension``
        projectors.
        """

        # cost function needs all the projectors
        if selected_level is None:
            selected_level = range(local_dimension)

            proj =  [
                ssp.csr_matrix(
                    ([1.0], ([ll], [ll])),
                    shape=(local_dimension, local_dimension)
                ).todia()
                for ll in selected_level
            ]

        # constraints need only projectors which violates 
        else:
            proj = ssp.csr_matrix(([1.0], ([selected_level],[selected_level])),
                            shape=(local_dimension,local_dimension))
            
        return proj
    

    # ---------------------------------------------------------------------- #
    #                            Instance Methods                            #
    # ---------------------------------------------------------------------- #
    def map(self, is_uniform=False, penalty = None, build_mode = None,  **kwargs):
        """
        Maps the MP4EO problem instance onto the corresponding
        many-qudit system basis.

        Arguments
        ---------
        is_uniform : bool, optional
            If :obj:`True`, the local dimensions of the qudits
            are padded to the largest local dimension among all
            the qudits.
            Default is :obj:`False`.

        **kwargs
            Additional optional arguments.
            If you need to rescale the couplings of the primal optimization
            problem cost function for numerical stability, you can provide
            an optional boolen argument ``rescale_primal_couplings``, whose
            default is set to :obj:`True`.

        Returns
        -------
        :obj:`None`
        """

        if build_mode is not None:
            self.set_build_mode(build_mode)

        if is_uniform:
            self._expand_local_basis()

        self.build_all_projectors(penalty=penalty)

        # Construct local observables 
        # self._construct_number_operators()

        # Construct cost function projectors
        # self._construct_dto_cost_func_projectors()
        # self._construct_dlo_cost_func_projectors()

        # Construct constraint projectors
        # self._construct_C3_projectors(penalty = penalty)
        # self._construct_C4_projectors(penalty = penalty)
        # self._construct_C5_projectors(penalty = penalty)
        # self._construct_C6_projectors(penalty = penalty)
        

    def _expand_local_basis(self):
        """
        Expands the local basis of both DTO and DLO qudits to reach the
        largest local dimension among all the qudits.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        Qudits in this mapping may have its own local dimension, i.e.,
        the :math:`j`-th qudit generally has :math:`D_j`-levels that may
        differ from the number of levels of another qudit in the system.
        It can be computationally required (but of course not convenient)
        to add dummy basis states in the Hilbert space of the individual
        qudit such that the dimension of the local basis :math:`D_j` is
        the same for all target index :math:`j = 0, \cdots, M - 1`, both
        for DTO and DLO sites.
        """

        DUMMY_LABEL = "PAD"

        # Get the largest local dimension (“fattest” qudit)
        max_local_dim = max(
            max(self._dto_local_dimensions), self._dlo_local_dimensions[0]
        )
        self._dto_local_dimensions = [max_local_dim] * self.n_targets
        self._dlo_local_dimensions = [max_local_dim] * self.n_targets

        # Expand DTO local basis via padding
        self._dto_local_basis = [
            np.pad(
                dto_local_basis,
                (0, max_local_dim - len(dto_local_basis)),
                constant_values=DUMMY_LABEL,
            )
            for dto_local_basis in self.dto_local_basis
        ]

        # Expand DLO local basis vi padding
        missing_logical_states = max_local_dim - len(self.dlo_local_basis[0])
        self._dlo_local_basis = [
            np.pad(
                dlo_local_basis,
                (0, missing_logical_states),
                constant_values=DUMMY_LABEL,
            )
            for dlo_local_basis in self._dlo_local_basis
        ]

    def _construct_number_operators(self):
        """
        Constructs the number operators for each DTO and DLO qudit
        which identifies the state of DTO or DLO.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        The DTO and DLO number operators are stored as lists where the
        :math:`j`-th element identifies the number operator for the
        :math:`j`-th target site.
        """

        # Define DTO number operators
        self._dto_number_projectors = {
            i : ssp.diags(diagonals=np.arange(Dj), format="csr")
            for i, Dj in enumerate(self._dto_local_dimensions)
        }

        # Define DLO number operators
        self._dlo_number_projectors = {
            i : ssp.diags(
                diagonals=np.arange(self._dlo_local_dimensions[0]),
                format="csr",
            )
            for i in range(self.n_targets)
        }

    def _construct_local_observables(self):
        """
        """
        pass

    def _NEW_dto_logical_state_idx_to_mp4eo_idx(self):
        """ 
        Builds a mapping from each logical state index of a DTO qudit to its corresponding
        mission indicese (i,j,k), where:
            -i: satellite index
            -j: target index
            -k: acquisition time slot or opportunity index

        
        The mapping is constructed for each target (DTO) and is stored in a list 
        of dictionaries, one per target.

        Returns
        -------
        qudit_state_idx_to_ijk_map: List[Dict[int, Tuple[int, int, int]]]
            List of dictionaries mapping each logical state index to the corresponding 
            mission triplet (i, j, k). One dictionary per target.
        """

        DUMMY_LABEL = "PAD"
        qudit_state_idx_to_ijk_map = []
        dto_local_dim = self._dto_local_dimensions

        for jj in range(self.n_targets):
            single_qudit_map_dict = {}
            ll = 0

            for ii in range(self.n_satellites):
                for kk in range(self._mp4eo_instance.thetaij(ii, jj)):
                    # if ll >= dto_local_dim[jj]:
                    #     break # stop physical states
                    single_qudit_map_dict[ll] = (ii,jj,kk)
                    ll +=1
            while ll < dto_local_dim[jj]:
                single_qudit_map_dict[ll] = (-1,-1,-1)
                ll += 1
            
            qudit_state_idx_to_ijk_map.append(single_qudit_map_dict)

            # assert ll == self._dto_local_dimensions[jj], MP4EOQuditsMapperError(f"WIERD MAPPING FROM QUDIT LOGICAL STATE TO MISSION TRIPLET OF INDICES")
            # qudit_state_idx_to_ijk_map.append(single_qudit_map_dict)
        # print('qudit state index:', qudit_state_idx_to_ijk_map)

        return qudit_state_idx_to_ijk_map

    def _NEW_dlo_logical_state_idx_to_mp4eo_idx(self):
        """ 
        Builds a mapping from each logical state index of a DLO qudit to its corresponding
        mission indicese (i,j,r), where:
            -i: satellite index
            -j: target index
            -r: acquisition time slot or opportunity index

        
        The mapping is constructed for each target (DLO) and is stored in a list 
        of dictionaries, one er target.

        Returns
        -------
        qudit_state_idx_to_ijr_map: List[Dict[int, Tuple[int, int, int]]]
            List of dictionaries mapping each logical state index to the corresponding 
            mission triplet (i, j, r). One dictionary per target.
        """
        qudit_state_idx_to_ijr_map = []

        for jj in range(self.n_targets):
            single_qudit_map_dict = {}

            ll = 0
            for ii in range(self.n_satellites):
                for rr in range(self._mp4eo_instance.sigma[ii]):
                    single_qudit_map_dict[ll] = (ii, jj, rr)
                    ll += 1

            assert ll == self._dlo_local_dimensions[jj], MP4EOQuditsMapperError("WIERD MAPPING FROM QUDIT LOGICAL STATE TO MISSION TRIPLET OF INDICES")
            qudit_state_idx_to_ijr_map.append(single_qudit_map_dict)
        return qudit_state_idx_to_ijr_map

    def _construct_dto_cost_func_projectors(self):
        """ 
        Constructs the cost function projectors acting on DTO qudits.

        For each DTO qudit (associated to a target j), constructs a local Hamiltonian
        as a linear combination of |l⟩⟨l| projectors, weighted by the corresponding
        primal cost coefficient w_{ijk} from the mission instance.

        The cost is optionally rescaled by a global factor.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Notes
        -----
        Uses the mapping (i, j, k) ← logical index l obtained from
        _NEW_dto_logical_state_idx_to_mp4eo_idx.

        Details
        -------
        The DTO projectors are stored as lists where the :math:`j`-th element identifies 
        the projector for the :math:`j`-th target site.
        """

        rescale_factor = self._primal_couplings_rescaling_factor
        dto_weight_mapper = self._NEW_dto_logical_state_idx_to_mp4eo_idx()
        self._dto_cost_function_projs = {}
       
        for jj, Dj in enumerate(self._dto_local_dimensions):
            local_projectors = self._get_single_qudit_projectors(Dj)
            ham_f = ssp.csr_matrix((Dj, Dj))

            for ll, projector in enumerate(local_projectors):
                proj_weight_idx = dto_weight_mapper[jj][ll]
                proj_weight = self._mp4eo_instance.wijk(proj_weight_idx[0], proj_weight_idx[1], proj_weight_idx[2]) / rescale_factor
                ham_f += proj_weight * projector

            self._dto_cost_function_projs[jj] = ham_f


    def _construct_dlo_cost_func_projectors(self):
        """
        Constructs the cost function projectors acting on DLO qudits.

        For each DLO qudit (associated to a target j), constructs a local Hamiltonian
        as a linear combination of |l⟩⟨l| projectors, weighted by the corresponding
        primal cost coefficient w_{ijr} from the mission instance.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Notes
        -----
        Uses the mapping (i, j, r) ← logical index l obtained from
        _NEW_dlo_logical_state_idx_to_mp4eo_idx.

        Details
        -------
        The DLO projectors are stored as lists where the :math:`j`-th element identifies 
        the projector for the :math:`j`-th target site.
        """
        rescale_factor = self._primal_couplings_rescaling_factor
        dlo_weight_mapper = self._NEW_dlo_logical_state_idx_to_mp4eo_idx()
        self._dlo_cost_function_projs = {}

        for jj, Dj in enumerate(self._dlo_local_dimensions):
                local_projectors = self._get_single_qudit_projectors(Dj)
                ham_f = ssp.csr_matrix((Dj, Dj))

                for ll, projector in enumerate(local_projectors):
                    proj_weight_idx = dlo_weight_mapper[jj][ll]
                    proj_weight = self._mp4eo_instance.wijr(proj_weight_idx[0], proj_weight_idx[1], proj_weight_idx[2]) / rescale_factor
                    ham_f += proj_weight * projector

                self._dlo_cost_function_projs[jj] = ham_f

    def _construct_C3_projectors(self, penalty):
        """ 
        Constructs the local DTO-DLO projectors to enforce constraint
        (3) between qudits on the same **mission physical site**.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        It is possible to set a specific penalty_weight, in order to 
        enforce or decrease the constraint weight in the full Hamiltonian.

        For each target index :math:`j = 1, \cdots, M`, this
        method constructs a projector that enforces mission
        planning constraint (3).

        Each projector is diagonal, and its entries are set to
        :math:`+1` if the corresponding DTO-DLO concatenated configuration
        **violates** the constraint, :math:`-1` if it satisfies the
        constraint.

        The projectors are stored in a list where the :math:`j`-th
        element identifies the projector for the :math:`j`-th target
        site.

        Local projectors are stored in a dict for TN solver.
        """
        # penalty_weight = 1.2
        Dj_y = self._dlo_local_dimensions[0]
        c3_projectors = {} if self._build_exact else None
        c3_for_TN = defaultdict(list)

        for jj, Dj_x in enumerate(self._dto_local_dimensions):

            # qudit indices for TN
            jx = 2*jj
            jy = 2*jj +1

            local_dim = Dj_x * Dj_y
            H_c3 = ssp.csr_matrix((local_dim, local_dim))
            for dto_level, dto_config in enumerate(self._dto_local_basis[jj]):
                dto_one_idx = dto_config.rfind("1")
                csum_arr = np.array([self._mp4eo_instance.thetaij(ii, jj) for ii in range(self.n_satellites)])
                dto_sat_idx = next(ii for ii, acc in enumerate(np.cumsum(csum_arr)) if dto_one_idx < acc)

                for dlo_level, dlo_config in enumerate(self._dlo_local_basis[jj]):
                    dlo_one_idx = dlo_config.rfind("1")
                    dlo_sat_idx = next(ii for ii, acc in enumerate(np.cumsum(self._mp4eo_instance.sigma)) if dlo_one_idx < acc)

                    if dto_sat_idx != dlo_sat_idx:
                        # |d_x⟩⟨d_x| \otimes |d_y⟩⟨d_y|
                        P_dto = self._get_single_qudit_projectors(Dj_x, dto_level)
                        P_dlo = self._get_single_qudit_projectors(Dj_y, dlo_level)
                        
                        if self._build_exact:
                            P_local = ssp.kron(P_dto, P_dlo, format='csr')
                            H_c3 += (penalty+0.1) * P_local
                        if self._build_tn:
                            c3_for_TN[(jx, jy)].append((dto_level, dlo_level, P_dto,P_dlo))
                        
            if self._build_exact:
                c3_projectors[jj] = H_c3

        self._C3_projectors = c3_projectors
        self._C3_projectors_for_TN = c3_for_TN

    def _construct_C4_projectors(self, penalty):
        """ 
        Constructs the local DTO-DLO projectors to enforce constraint
        (4) between qudits on the same **mission physical site**.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        :obj:`None`

        Details
        -------
        It is possible to set a specific penalty_weight, in order to 
        enforce or decrease the constraint weight in the full Hamiltonian.

        For each target index :math:`j = 1, \cdots, M`, this
        method constructs a projector that enforces mission
        planning constraint (4), leveraging the pre-processing
        step at the level of the instance class.

        Each projector is diagonal, and its entries are set to
        :math:`+1` if the corresponding DTO-DLO concatenated configuration
        **violates** the constraint, :math:`0` if it satisfies the
        constraint.

        The projectors are stored in a dictonary where the key :math:`j`
        identifies the projector for the :math:`j`-th target site.

        Local projectors are stored in a dict for TN solver.
        """
        # penalty_weight = 1.2
        Dj_y = self._dlo_local_dimensions[0]
        c4_projectors = {} if self._build_exact else None
        c4_for_TN = defaultdict(list)

        for jj, Dj_x in enumerate(self._dto_local_dimensions):

            # qudit indices for TN
            jx = 2*jj
            jy = 2*jj +1

            local_dim = Dj_x * Dj_y
            H_c4 = ssp.csr_matrix((local_dim, local_dim))
            for dto_level, dto_config in enumerate(self._dto_local_basis[jj]):
                dto_one_idx = dto_config.rfind("1")
                csum_arr_dto = np.array([self._mp4eo_instance.thetaij(ii, jj) for ii in range(self.n_satellites)])
                dto_sat_idx = next(ii for ii, acc in enumerate(np.cumsum(csum_arr_dto)) if dto_one_idx < acc)

                for dlo_level, dlo_config in enumerate(self._dlo_local_basis[jj]):
                    dlo_one_idx = dlo_config.rfind("1")
                    csum_arr_dlo = np.array([self._mp4eo_instance.sigma[ii] for ii in range(self.n_satellites)])
                    dlo_sat_idx = next(ii for ii, acc in enumerate(np.cumsum(csum_arr_dlo)) if dlo_one_idx < acc)

                    if dto_sat_idx == dlo_sat_idx:
                        dto_kk = dto_one_idx - (
                            csum_arr_dto[dto_sat_idx - 1] if dto_sat_idx > 0 else 0
                        )
                        dlo_rr = dlo_one_idx - (
                            csum_arr_dlo[dlo_sat_idx - 1] if dlo_sat_idx > 0 else 0
                        )

                        if (dto_kk, dlo_rr) in self._mp4eo_instance.get_preprocessed_c4_idxs(dto_sat_idx, jj):
                            # |d_x⟩⟨d_x| \otimes |d_y⟩⟨d_y|
                            P_dto = self._get_single_qudit_projectors(Dj_x, dto_level)
                            P_dlo = self._get_single_qudit_projectors(Dj_y, dlo_level)
                            if self._build_exact:
                                P_local = ssp.kron(P_dto, P_dlo, format='csr')
                                H_c4 += (penalty + 0.1) * P_local
                            if self._build_tn:
                                c4_for_TN[(jx, jy)].append((dto_level, dlo_level, P_dto,P_dlo))
            
            if self._build_exact:                
                c4_projectors[jj] = H_c4
        self._C4_projectors = c4_projectors
        self._C4_projectors_for_TN = c4_for_TN

    def _construct_C5_projectors(self, penalty):
        """ 
        Constructs and embeds the global operator enforcing constraint (C5) 
        on DTO-DTO pairs associated with the same satellite and different 
        **mission physical sites**.

        Arguments
        ---------
        None

        Returns
        -------
        None

        Details
        -------
        This method constructs an operator that enforces mission
        planning constraint (5), leveraging the pre-processing
        step at the level of the instance class.

        Each operator is diagonal, and its entries are penalized by
        :math:`+1` every time the corresponding DTO-DTO concatenated
        configuration **violates** the constraint, :math:`0` if it
        satisfies the constraint.

        Details
        -----
        - Only configurations violating the constraint contribute to 
        the Hamiltonian, with a positive penalty (default = 1.1).
        - The final operator is directly constructed in its global 
        (fully tensorized) form.
        - The result is a single sparse matrix representing the 
        complete C5 penalty term, ready to be included in the global Hamiltonian.
        - Local projectors before tensorization are stored in a dict for TN solver.
        """
        # penalty_weight = 1.1
        H_c5 = None
        c5_for_TN = defaultdict(list)
        if self._build_exact:
            H_c5 = ssp.csr_matrix((self.hilbert_space_dim, self.hilbert_space_dim))
            local_dims = self.local_dimensions
        
        for jj, Dj_x in enumerate(self._dto_local_dimensions):
            for ss, Ds_x in enumerate(self._dto_local_dimensions):
                if jj == ss: 
                    continue
                else:

                    # qudit indices for TN
                    jx = 2*jj
                    sx = 2*ss

                    for dto_j_level, dto_j_config in enumerate(self._dto_local_basis[jj]):
                        dto_one_idx_kk = dto_j_config.rfind("1")
                        csum_arr_j = np.array([self._mp4eo_instance.thetaij(ii, jj) for ii in range(self.n_satellites)])
                        dto_j_sat_idx = next(ii for ii, acc in enumerate(np.cumsum(csum_arr_j)) if dto_one_idx_kk < acc)

                        for dto_s_level, dto_s_config in enumerate(self._dto_local_basis[ss]):
                            dto_one_idx_kp = dto_s_config.rfind("1")
                            csum_arr_s = np.array([self._mp4eo_instance.thetaij(ii, ss) for ii in range(self.n_satellites)])
                            dto_s_sat_idx = next(ii for ii, acc in enumerate(np.cumsum(csum_arr_s)) if dto_one_idx_kp < acc)

                            if dto_j_sat_idx == dto_s_sat_idx:
                                dto_kk = dto_one_idx_kk - (csum_arr_j[dto_j_sat_idx - 1] if dto_j_sat_idx > 0 else 0)
                                dto_kp = dto_one_idx_kp - (csum_arr_s[dto_s_sat_idx - 1] if dto_s_sat_idx > 0 else 0)
                                
                                # exact
                                if (jj, ss, dto_kk, dto_kp) in self._mp4eo_instance.get_preprocessed_c5_idxs(dto_j_sat_idx):
                                    # |dj_x⟩⟨dj_x| \otimes |ds_y⟩⟨ds_y|
                                    Pj_dto = self._get_single_qudit_projectors(Dj_x, dto_j_level)
                                    Ps_dto = self._get_single_qudit_projectors(Ds_x, dto_s_level)
                                    if self._build_exact:
                                        H_c5 += penalty * embed_two_body(Pj_dto, Ps_dto, local_dims, "DTO", jj, ss)
                                    
                                    if self._build_tn:
                                        c5_for_TN[(jx,sx)].append((dto_j_level, dto_s_level, Pj_dto, Ps_dto))
                
        self._C5_projectors = H_c5
        self._C5_projectors_for_TN = c5_for_TN
    
    def _construct_C6_projectors(self, penalty):
        """ 
        Constructs and embeds the global operator enforcing constraint (C6) 
        on DTO-DTO pairs associated with the same satellite and different 
        **mission physical sites**.

        Arguments
        ---------
        None

        Returns
        -------
        None

        Details
        -------
        This method constructs an operator that enforces mission
        planning constraint (6), leveraging the pre-processing
        step at the level of the instance class.

        Each operator is diagonal, and its entries are penalized by
        :math:`+1` every time the corresponding DTO-DTO concatenated
        configuration **violates** the constraint, :math:`0` if it
        satisfies the constraint.

        Details
        -----
        - Only configurations violating the constraint contribute to 
        the Hamiltonian, with a positive penalty (default = 1.1).
        - The final operator is directly constructed in its global 
        (fully tensorized) form.
        - The result is a single sparse matrix representing the 
        complete C6 penalty term, ready to be included in the global Hamiltonian.
        - Local projectors before tensorization are stored in a dict for TN solver.
        """

        c6_for_TN = defaultdict(list)
        H_c6 = None
        if self._build_exact:
            H_c6 = ssp.csr_matrix((self.hilbert_space_dim, self.hilbert_space_dim))
            local_dims = self.local_dimensions

        for jj, Dj_y in enumerate(self._dlo_local_dimensions):
            for gg, Dg_y in enumerate(self._dlo_local_dimensions):
                if jj == gg:
                    continue

                else:

                    jx = jj*2 + 1
                    sx = gg*2 + 1

                    for dlo_j_level, dlo_j_config in enumerate(self._dlo_local_basis[jj]):
                        dlo_one_idx_rr = dlo_j_config.rfind("1")
                        csum_arr_j = np.array([self._mp4eo_instance.sigma[ii] for ii in range(self.n_satellites)])
                        dlo_j_sat_idx = next(ii for ii, acc in enumerate(np.cumsum(csum_arr_j)) if dlo_one_idx_rr < acc)

                        for dlo_g_level, dlo_g_config in enumerate(self._dlo_local_basis[gg]):
                            dlo_one_idx_rp = dlo_g_config.rfind("1")
                            csum_arr_g = np.array([self._mp4eo_instance.sigma[ii] for ii in range(self.n_satellites)])
                            dlo_g_sat_idx = next(ii for ii, acc in enumerate(np.cumsum(csum_arr_g)) if dlo_one_idx_rp < acc)

                            if dlo_j_sat_idx == dlo_g_sat_idx:
                                dlo_rr = dlo_one_idx_rr - (csum_arr_j[dlo_j_sat_idx - 1] if dlo_j_sat_idx > 0 else 0)
                                dlo_rp = dlo_one_idx_rp - (csum_arr_g[dlo_g_sat_idx - 1] if dlo_g_sat_idx > 0 else 0)

                                if (jj, gg, dlo_rr, dlo_rp) in self._mp4eo_instance.get_preprocessed_c6_idxs(dlo_j_sat_idx):
                                    # |dj_y⟩⟨dj_y| \otimes |dg_y⟩⟨dg_y|
                                    Pj_dlo = self._get_single_qudit_projectors(Dj_y, dlo_j_level)
                                    Pg_dlo = self._get_single_qudit_projectors(Dg_y, dlo_g_level)
                                    if self._build_exact:
                                        H_c6 += penalty * embed_two_body(Pj_dlo, Pg_dlo, local_dims, "DLO", jj, gg)
                                    if self._build_tn:
                                        c6_for_TN[(jx,sx)].append((dlo_j_level, dlo_g_level, Pj_dlo, Pg_dlo))
                
        self._C6_projectors = H_c6
        self._C6_projectors_for_TN = c6_for_TN

    def join_mission_sites(self):
        """
        Joins the local dimensions, cost function projectors and number operators
        for each DTO and DLO qudit into unified lists that represent the full
        Hilbert space of the system.

        Arguments
        ---------
        :obj:`None`

        Returns
        -------
        local_dimensions : np.ndarray[int]
            The local dimensions per physical site.

        joined_cost_func_projectors : np.ndarray[scipy.sparse.csr_matrix]
            The list of local cost function projectors (sparse matrices), one for each
            physical site

        joined_number_operators : np.array[scipy.sparse.csr_matrix]
            The list of number operators (sparse matrices), one for each physical site,
            ordered in the same way as the projectors.
        Notes
        -----
        Each physical site alternates between a DTO qudit and a DLO qudit.
        """

        # Join local dimensions
        local_dims = [
            item
            for pair in zip(
                self._dto_local_dimensions, self._dlo_local_dimensions
            )
            for item in pair
        ]

        joined_cost_func_projectors = [] 
        for key in self._dto_cost_function_projs.keys():
            joined_cost_func_projectors.append(self._dto_cost_function_projs[key])
            joined_cost_func_projectors.append(self._dlo_cost_function_projs[key])

        joined_number_operators = [] 
        for key in self._dto_number_projectors.keys():
            joined_number_operators.append(self._dto_number_projectors[key])
            joined_number_operators.append(self._dlo_number_projectors[key])

        # Output
        return local_dims, joined_cost_func_projectors, joined_number_operators

    def qudit_occupations_to_mission_bitstring(self, qudit_occupations):
        """
        Retrieves the MP4EO bit-string from the many-qudit
        system occupation numbers.

        Arguments
        ---------
        qudit_occupations : List[int]
            The projections onto the logical states of the final
            many-qudit state after an optimization.

        Returns
        -------
        mission_bitstring : np.ndarray[int]
            The reconstructed MP4EO bit-string that defines the planned
            mission.

        Details
        -------
        Occupation numbers for each DTO and DLO qudit
        in the mapped many-body system are measured on
        the final state within the specific quantum-inspired
        solver in :mod:`mp4eo_optimization.mp4eo_solvers`.
        Once the occupations for each qudit are known, i.e.,
        once the logical state of each qudit on the many-body
        state is known, we can reconstruct the original MP4EO
        bit-string by concatenating the bit-strings defining
        the logical qudit basis states stored in
        :attr:`~QuditsMapper._dto_local_basis`
        and :attr:`~QuditsMapper._dlo_local_basis`.

        Notes
        -----
        We have :math:`2M` qudits in the mapped system, two for
        each target :math:`j = 0, \cdots, M` to acquire: one DTO
        qudit and one DLO qudit.
        This means we have :math:`M` **mission-physical site**, but
        in practice, they correspond to :math:`2M` physical sites.
        The ``qudit_occupations`` input list of projections on the
        computational basis alternates between a DTO qudit and a DLO
        qudit for each mission site :math:`j`.
        As a result, the even-indexed elements represent the occupation
        numbers :math:`\hat{N}^{(j)}_x`of the DTO qudits, while the
        odd-indexed elements correspond to the DLO qudits
        :math:`\hat{N}^{(j)}_y`.
        """

        # Split DTO and DLO energy-level occupations
        dto_occupations = [round(Nx) for Nx in qudit_occupations[::2]]
        dlo_occupations = [round(Ny) for Ny in qudit_occupations[1::2]]

        # Read logical state-related bit-strings and concatenate
        mission_bitstring = "".join(
            self.get_jth_dto_logical_states(jj)[Nx]
            + self.get_jth_dlo_logical_states(jj)[Ny]
            for jj, (Nx, Ny) in enumerate(
                zip(dto_occupations, dlo_occupations)
            )
        )
        # FIXME: when padding with dummy states, how do you
        # FIXME: concatenate back strings? Handle the DUMMY_LABEL case !!!!
        # FIXME: this is an hidden bug!

        return self._rearrange_mission_bitstring(
            np.fromiter(mission_bitstring, dtype=np.uint8)
        )

    def get_jth_dto_logical_states(self, dto_site):
        """
        Gets the local basis for the :math:`j`-th DTO qudit.

        Arguments
        ---------
        dto_site : int
            The index of the DTO qudit site :math:`j`.

        Returns
        -------
        :obj:`None`
        """

        assert 0 <= dto_site < self.n_targets, MP4EOQuditsMapperError(
            "Invalid DTO site index."
        )
        return self.dto_local_basis[dto_site]

    def get_jth_dlo_logical_states(self, dlo_site):
        """
        Gets the local basis for the :math:`j`-th DLO qudit.

        Arguments
        ---------
        dlo_site : int
            The index of the DLO qudit site :math:`j`.

        Returns
        -------
        :obj:`None`
        """

        assert 0 <= dlo_site < self.n_targets, MP4EOQuditsMapperError(
            "Invalid DLO site index."
        )
        return self.dlo_local_basis[dlo_site]

    def _rearrange_mission_bitstring(self, mission_bitstring):
        """
        Re-arranges the MP4EO bit-string retrieved from the many-qudit
        state to match the binary order in the original MP4EO problem.

        Arguments
        ---------
        mission_bitstring : np.ndarray[int]
            The MP4EO bit-string retrieved from the optimized and
            measured many-qudit system.

        Returns
        -------
        rearranged_bitstring : np.ndarray[int]
            The re-arranged MP4EO bit-string.

        Notes
        -----
        This implementation is based on how the logical states of
        the qudits are defined and how the binary variables of the
        MP4EO problem are arranged within the bit-string segment
        that represents each qudit level.
        """

        rearranged_bitstring = np.zeros(
            self._mp4eo_instance.n_mp4eo_decision_variables, dtype=np.uint8
        )

        # Re-arrangement based on the binary ordering
        idx = 0
        for jj in range(self.n_targets):
            for ii in range(self.n_satellites):
                for kk in range(self._mp4eo_instance.thetaij(ii, jj)):
                    re_idx = self._mp4eo_instance._dto_idxs_to_sg_idx(
                        sat_idx=ii, tar_idx=jj, dto_idx=kk
                    )
                    rearranged_bitstring[re_idx] = mission_bitstring[idx]
                    idx += 1
            for ii in range(self.n_satellites):
                for rr in range(self._mp4eo_instance.sigma[ii]):
                    re_idx = self._mp4eo_instance._dlo_idxs_to_sg_idx(
                        sat_idx=ii, tar_idx=jj, dlo_idx=rr
                    )
                    rearranged_bitstring[re_idx] = mission_bitstring[idx]
                    idx += 1

        # Output
        return rearranged_bitstring
