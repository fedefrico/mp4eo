#############################################################################
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
Simulation Utilities Submodule
==============================

This submodule provides utilities for simulations.
The utilities allow for customizable configuration.
"""


##############################################################################
#                                  MODULES                                   #
##############################################################################

# Standard library
from functools import wraps
import random
from socket import gethostname
import scipy.sparse as ssp

# Third-party libraries
import numpy as np

# First-party libraries
from mp4eo_optimization.version import __version__

__all__ = [
    "print_preamble",
    "print_epilogue",
    "add_mp4eo_instance_cli_options",
    "add_solver_cli_options",
    "add_tn_cli_options",
    "postprocess_cli_options",
    "embed_one_body",
    "embed_two_body",
]

##############################################################################
##############################################################################


def ensure_init(attr):
    """
    Defines decorator for non-initialized class attributes.

    Arguments
    ---------
    attr : str
        String representing the name of the class attribute
        to be checked.

    Returns
    -------
    decorator : Callable
        The decorator to ensure if ``attr`` has been initialized.

    Raises
    ------
    :exc:`AttributeError`
        If class instance attribute has not been initialized.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, attr) is None:
                raise AttributeError(
                    f"The {attr} attribute has not been initialized."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def qubits_to_binaries(qubit_bitstring):
    r"""
    Qubit-to-binary converter.

    Arguments
    ---------
    qubit_bitstring : List[int] | np.ndarray
        The string of +1 and -1 representing the
        quantum expectation value of the Z-Pauli
        matrix on each site (+1 means spin-up while
        -1 mean spin-down).

    Returns
    -------
    bitstring : np.ndarray[int]
        The converted string of 0s and 1s.

    Raises
    ------
    :exc:`ValueError`
        If invalid spin states are found.

    Details
    -------
    Implements the conversion from qubit (spin-:math:`1 \mathbin{/} 2`)
    variables to binary variables, e.g., the decision variables of a
    combinatorial optimization problem.
    The applied linear transformation is

    .. math::

        x = \dfrac{1 + \sigma}{2}
        x \in \left\{0, 1\right\} (binary)
        \sigma \in \left\{-1, +1\right\} (qubit)

    that maps bit :math:`x = 0` into spin-down and bit :math`x = 1`
    into spin-up.
    """

    # Read the input qubit state
    bitstring = np.asarray(qubit_bitstring, dtype=int)

    # Check non-valid values
    if not np.all(np.isin(bitstring, [-1, 1])):
        raise ValueError(
            "Invalid measurement: spin-1/2 state must be up or down."
        )

    # Output
    return (bitstring + 1) // 2


def print_preamble(mpi_comm=None):
    """
    Prints code information common to all the simulations
    of this library to the standard output.

    Arguments
    ---------
    mpi_comm : mpi4py.MPI.Comm, optional
        The MPI communicator to be used.
        Default to :obj:`None`, meaning sequential execution.

    Returns
    -------
    :obj:`None`

    Details
    -------
    In MPI simulations, prints to standard output only on the
    0-rank process.
    """

    # Print system and code version information
    if mpi_comm is None or mpi_comm.Get_rank() == 0:
        print("*** START OF THE SIMULATION ***")
        space = "   "
        sep_line_1 = "-" * len("*** START OF THE SIMULATION ***")

        print(sep_line_1)
        print(f"\t Version {__version__}")
        print(sep_line_1)

        print(f"\n\n\nRunning on {gethostname()}")

        print("\n\n")

    # Set random seeds for reproducibility
    random.seed(0)
    np.random.seed(0)


def print_epilogue(tot_runtime, mpi_comm=None):
    """
    Prints final information common to all the simulations
    of this library to the standard output.

    Arguments
    ---------
    tot_runtime : float
        The total CPU running time of the whole
        simulation in seconds.

    mpi_comm : mpi4py.MPI.Comm, optional
        The MPI communicator to be used.
        Default to :obj:`None`, meaning sequential execution.

    Returns
    -------
    :obj:`None`

    Details
    -------
    In MPI simulations, prints to standard output only on the
    0-rank process.
    """

    # Print final information
    if mpi_comm is None or mpi_comm.Get_rank() == 0:
        print("\n\n")
        sep_line_1 = "-" * len("*** END OF THE SIMULATION ***")

        print("The simulation has successfully completed")

        print(sep_line_1)
        print(f"   CPU time: {tot_runtime:.2f} seconds")
        print(f"   CPU time: ~ {int(tot_runtime // 86400)} days")
        print(sep_line_1)

        print("*** END OF THE SIMULATION ***")


# Public release modification (2026-11-19): remove industrial range preset from CLI help.
def add_mp4eo_instance_cli_options(parser):
    """
    Adds command-line interface (CLI) options for the
    MP4EO problem to the given ``parser`` for use within
    all the simulations of this library.

    Arguments
    ---------
    parser : argparse.ArgumentParser
        The parser to which the MP4EO problem options are added.

    Returns
    -------
    :obj:`None`
    """

    # Define group of required options for the MP4EO problem
    required_group = parser.add_argument_group(
        "required arguments for the MP4EO problem"
    )

    # Add CLI options for the MP4EO problem
    required_group.add_argument(
        "-N",
        "--number_of_satellites",
        type=int,
        help="The number of EO satellites in the constellation.",
        required=True,
    )
    required_group.add_argument(
        "-M",
        "--number_of_acquisition_requests",
        type=int,
        help="The number of acquisition requests (EO targets) from the user.",
        required=True,
    )

    # MP4EO instance generation
    parser.add_argument(
        "-ndto",
        "--number_of_data_take_opportunities",
        type=int,
        help=" ".join(
            [
                "Request a custom number of DTO binaries in the problem using",
                "this option. The provided value will be further processed and",
                "if necessary, adjusted based on the parameters controlling the",
                "number of satellites and targets. This ensures consistency",
                "with the problem formulation as a linear integer program.",
                "Default to `None`, i.e., the MP4EO instance will have a random",
                "number of DTO binaries.",
            ]
        ),
        default=None,
    )
    parser.add_argument(
        "-ndlo",
        "--number_of_downlink_opportunities",
        type=int,
        help=" ".join(
            [
                "Request a custom number of DLO binaries in the problem using",
                "this option. The provided value will be further processed and",
                "if necessary, adjusted based on the parameters controlling the",
                "number of satellites and targets. This ensures consistency",
                "with the problem formulation as a linear integer program.",
                "Default to `None`, i.e., the MP4EO instance will have a random",
                "number of DLO binaries.",
            ]
        ),
        default=None,
    )
    parser.add_argument(
        "-type",
        "--problem_type",
        type=str,
        choices=["synthetic", "test"],
        help=" ".join(
            [
                "Type of MP4EO problem instance to be generated.",
                "This option controls the ranges for generating",
                "parameter values of a random MP4EO instance.",
                "The available options are `test`, which generates",
                "small random instances for code testing;",
                "Parameter ranges must be independently specified",
                "by the caller; no industrial presets are bundled.",
                "Default to `test`.",
            ]
        ),
        default="test",
    )
    parser.add_argument(
        "-init_mode",
        "--instance_initialization_mode",
        type=str,
        choices=["random-uniform", "from_file"],
        help=" ".join(
            [
                "Mode option for the MP4EO problem instance initialization.",
                "The available options are as follows:",
                "`random-uniform` generates a random instance where all the",
                "problem parameters are drawn from a uniform distribution",
                "within independently specified synthetic ranges;",
                "`from_file` reads the instance from an input file describing",
                "the MP4EO instance to be solved.",
                "Default to `random-uniform`.",
            ]
        ),
        default="random-uniform",
    )
    parser.add_argument(
        "-instance_input_file",
        "--instance_input_file",
        type=str,
        help=" ".join(
            [
                "The full path of the input file containing the parameters",
                "for the MP4EO problem instance when the initialization mode",
                "is set to `from_file`.",
                "Default to None, i.e., no input file will be used for",
                "initializing the problem instance.",
            ]
        ),
        default=None,
    )
    parser.add_argument(
        "-binaries_arrangement",
        "--binaries_arrangement",
        type=str,
        choices=["target-wise", "satellite-wise", "type-wise"],
        help=" ".join(
            [
                "The order in which the binary optimization variables",
                "appear in the bit-string describing a mission.",
                "Several orders are available:",
                "`satellite-wise` introduces all the variables related",
                "to each individual satellite in ascending order of target,",
                "first the DTOs and then the DLOs, with the satellite index",
                "being the slowest varying index.",
                "`target-wise` introduces all the binaries related to the same",
                "target, with the satellite index changing in increasing order,",
                "first the DTOs and then the DLOs."
                "`type-wise` introduces all the DTOs variables first, with the",
                "target index varying more slowly than the satellite one.",
                "Default to `target-wise`, i.e., the target index is the slowest",
                "varying one.",
            ]
        ),
        default="target-wise",
    )

    # Linear integer program constraints
    for ii in range(1, 8):
        ## Constraints to be considered
        parser.add_argument(
            f"+c{ii}",
            f"++constraint_{ii}_flag",
            help=" ".join(
                [
                    "Flag indicating whether to consider (True) or",
                    f"not (False) constraint ({ii}) during the optimization.",
                    "If specified as a command-line option,",
                    f"constraint {ii}) will be considered during the",
                    "optimization.",
                ]
            ),
            action="store_true",
        )

        ## Correspoonding “importance“ weights in the cost function
        parser.add_argument(
            f"-lambda{ii}_ratio",
            f"--constraint_{ii}_qubo_penalty_constant_ratio",
            type=float,
            help=" ".join(
                [
                    "The value of the ratio between",
                    f"the penalty constant $lambda_{ii}$",
                    f"for enforcing constraint {ii}) in the",
                    "QUBO cost function and the maximum",
                    "absolute value of the weights defining",
                    "the MP4EO problem instance cost function.",
                    "Default to 1.5.",
                ]
            ),
            default=1.5,
        )


def add_solver_cli_options(parser):
    """
    Adds command-line interface (CLI) options for the
    MP4EO-problem solver to the given ``parser`` for
    use within all the simulations of this library.

    Arguments
    ---------
    parser : argparse.ArgumentParser
        The parser to which the MP4EO-problem solver options
        are added.

    Returns
    -------
    :obj:`None`
    """

    # --------------------------------------------------- #
    #               MP4EO problem optimizer               #
    # --------------------------------------------------- #
    parser.add_argument(
        "-optimizer",
        "--optimizer",
        type=str,
        choices=[
            "cplex",
            "gurobi",
            "exact-diagonalization",
            "tensor-network",
            "qudit",
        ],
        help=" ".join(
            [
                "The optimizer used to find the optimal solution to the input",
                "MP4EO problem instance.",
                "Several optimizers are available:",
                "`cplex` uses the classical Docplex solver for quadratic",
                "programs (free license are limited in the number of binaries);",
                "`gurobi` uses the Gurobi Optimizer suite for solving QUBO",
                "problems (free license are limited in the number of binaries);",
                "`exact-diagonalization` is an exact quantum-inspired solver",
                "that finds the ground state of a corresponding many-body",
                "Hamiltonian encoding problem's solutions (limited number of",
                "binaries due to the explosion of the Hilbert space dimension);",
                "`tensor-network` performs the same ground-state search",
                "with Tensor Networks, thus scaling up the number of optimization",
                "variables.",
                "Default to `cplex`.",
            ]
        ),
        default="cplex",
    )
    parser.add_argument(
        "-timelimit",
        "--timelimit",
        type=float,
        help=" ".join(
            [
                "A limit in seconds on the elapsed time (wall clock)",
                "for the solver to find a solution.",
                "Default to 1e75, i.e., de-facto no-time limit imposed.",
            ]
        ),
        default=1e75,
    )
    parser.add_argument(
        "-n_solutions",
        "--target_number_of_solutions",
        type=int,
        help=" ".join(
            [
                "The target number of solutions to compute",
                "through the chosen MP4EO solver.",
                "Each solution corresponds to a planned mission.",
                "Default to 1, i.e., only the optimal solution",
                "will be computed.",
            ]
        ),
        default=1,
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        help="Random seed for the solver.",
        default=0,
    )

    # --------------------------------------------------- #
    #              Quantum-inspired solvers               #
    # --------------------------------------------------- #
    parser.add_argument(
        "-qmapping",
        "--quantum_mapping",
        type=str,
        choices=["qubits", "qudits"],
        help=" ".join(
            [
                "The mapping to be used to transform the original binary",
                "optimization problem into one involving quantum degrees of",
                "freedom, then treatable by a quantum or quantum-inspired",
                "method.",
                "Two types of mappings are available:",
                "`qubits` maps each optimization binary into a qubit",
                "(spin glass mapping);",
                "`qudits` maps groups of binaries into appropriate D-level",
                "systems.",
                "Default mapping to `qubits`.",
            ]
        ),
        default="qubits",
    )


def add_tn_cli_options(parser):
    """
    Adds command-line interface (CLI) options for the tensor
    network implementation to the given ``parser`` for use
    within all the simulations of this library.

    Arguments
    ---------
    parser : argparse.ArgumentParser
        The parser to which the tensor-network options are added.

    Returns
    -------
    :obj:`None`
    """

    # Ansatz
    parser.add_argument(
        "-tn_topology",
        "--tensor_network_topology",
        type=str,
        choices=["MPS", "TTN"],
        help=" ".join(
            [
                "The tensor network topology.",
                "Available topologies are",
                "Tree Tensor Network (TTN), and",
                "Matrix Product States (MPS).",
                "Default to TTN.",
            ]
        ),
        default="TTN",
    )
    parser.add_argument(
        "-m",
        "--max_bond_dimension",
        type=int,
        help="The maximum bond dimension of the tensor network.",
        default=8,
    )
    parser.add_argument(
        "-min_bdim",
        "--min_bond_dimension",
        type=int,
        help=" ".join(
            [
                "The minimum bond dimension of the tensor network.",
                "Default to None, i.e., same value of the maximum one.",
            ]
        ),
        default=None,
    )
    parser.add_argument(
        "-cut_ratio",
        "--cut_ratio",
        type=float,
        help="Cut ratio for the singular values truncation.",
        default=1e-9,
    )
    parser.add_argument(
        "-trunc_method",
        "--truncation_method",
        type=str,
        help="Method used to truncate the singular values.",
        default="R",
    )

    # Initial tensor-network state
    parser.add_argument(
        "-initial_tn_state",
        "--initial_tensor_network_state",
        type=str,
        choices=["random", "driver_product_state", "haar"],
        help=" ".join(
            [
                "The initial tensor network state for starting",
                "the ground-state search. The available options are:",
                "`random` initialization, `driver_product_state` to",
                "initialize the tensors in the product state of the",
                "driver Hamiltonian, and `haar` to initialize the tensors",
                "in a product state of local Haar state.",
                "Default to `random`.",
            ]
        ),
        default="random",
    )
    parser.add_argument(
        "-init_bond_dimension",
        "--initial_state_bond_dimension",
        type=int,
        help=" ".join(
            [
                "The bond dimension of the initial tensor-network state.",
                "Default to None, i.e., the initial-state bond dimension",
                "is equal to the maximum bond dimension.",
            ]
        ),
        default=None,
    )
    parser.add_argument(
        "-padding_noise",
        "--initial_state_padding_noise",
        type=float,
        help=" ".join(
            [
                "Small padding noise to ensure a non-trivial",
                "bond dimension to the chosen initial tensor",
                "network state. Default to 1e-7.",
            ]
        ),
        default=1e-7,
    )

    # Ground-state search
    parser.add_argument(
        "-nsweeps",
        "--number_of_sweeps",
        type=int,
        help=" ".join(
            [
                "The maximum number of sweeps for the ground-state search.",
                "After this number of sweeps the ground-state search ends.",
                "Default to 20.",
            ]
        ),
        default=20,
    )
    parser.add_argument(
        "-abs_dev",
        "--absolute_deviation",
        type=float,
        help=" ".join(
            [
                "Exit criterion for the ground-state search.",
                "If the energy of the current sweep has an",
                "absolute per-sweep deviation from the previous",
                "sweep below this threshold, the optimization ends.",
                "Default to 4e-12.",
            ]
        ),
        default=4e-12,
    )
    parser.add_argument(
        "-rel_dev",
        "--relative_deviation",
        type=float,
        help=" ".join(
            [
                "Exit criterion for the ground-state search.",
                "If the energy of the current sweep has an",
                "relative per-sweep deviation from the previous",
                "sweep below this threshold, the optimization ends.",
                "Default to 1e-12.",
            ]
        ),
        default=1e-12,
    )
    parser.add_argument(
        "-n_points_conv_check",
        "--number_of_convergence_check_points",
        type=int,
        help=" ".join(
            [
                "The number of sweeps used when checking convergence for the",
                "ground-state search. Exit criteria are not checked until this",
                "many sweeps have been performed. If equal to the number of",
                "sweeps, the convergence criteria will never be checked.",
                "Default to 6 sweeps.",
            ]
        ),
        default=6,
    )
    parser.add_argument(
        "-tfields_ratio",
        "--transverse_fields_ratio",
        type=float,
        help=" ".join(
            [
                "The ratio controlling the order of magnitude of",
                "the transverse fields as perturbation to escape",
                "local minima.",
                "Default to None.",
            ]
        ),
        default=None,
    )
    parser.add_argument(
        "-do_random_sweeps",
        "--do_random_sweeps",
        help="If specified, perform random sweeps during the ground-state search",
        action="store_true",
    )
    parser.add_argument(
        "-arnoldi_maxiter",
        "--arnoldi_maxiter",
        type=int,
        help=" ".join(
            [
                "Maximum number of iterations in the arnoldi method used",
                "to solve the local effective eigenvalue problem within",
                "the DMRG-like optimization.",
                "Default to 32.",
            ]
        ),
        default=32,
    )
    parser.add_argument(
        "-do_spacelink_expansion",
        "--do_spacelink_expansion",
        help=" ".join(
            [
                "If specified, the first half of the sweeps",
                "uses space-link expansion, while in the second",
                "half the space-link expansion is disabled.",
            ]
        ),
        action="store_true",
    )

    # Tensors level
    parser.add_argument(
        "-tn_dtype",
        "--tensors_data_type",
        type=str,
        choices=["Z", "C", "D", "S"],
        nargs="+",
        help=" ".join(
            [
                "Numerical data type of the tensors. Default to Z.",
                "You can pass a sequence of characters to change the",
                "data type for each sweep.",
            ]
        ),
        default="Z",
    )
    parser.add_argument(
        "-tn_dtype_mode",
        "--tensors_data_type_mode",
        type=str,
        choices=["homogeneous", "half-increasing", "half-decreasing"],
        help=" ".join(
            [
                "Control option for the numerical data type of the tensors",
                "during the ground-state search.",
                "Once a data type is selected using the previous CLI option,",
                "it can be used consistently for all sweeps (`homogeneous`),",
                "or the sweeps can be split, using the same chosen data type",
                "for the first half and then either increasing",
                "(`half-increasing`) or decreasing (`half-decreasing`) the",
                "numerical precision for the second half.",
                "Default to `homogeneous` data-type throughout the sweeps.",
            ]
        ),
        default="homogeneous",
    )
    parser.add_argument(
        "-tn_btype",
        "--tensors_backend_type",
        type=str,
        choices=["numpy", "torch"],
        help=" ".join(
            [
                "The linear algebra backend to be used",
                "for low-level tensor operations.",
            ]
        ),
        default="numpy",
    )
    parser.add_argument(
        "-mpo_opt_level",
        "--mpo_representation_optimization_level",
        type=int,
        choices=[-1, 0, 1, 3, 10, 11],
        help=" ".join(
            [
                "Optimization level for the MPO representation.",
                "See qtealeaves or the convergence parameters submodule",
                "of this project for more details.",
                "Default to -1, which enables an auto-selection.",
                "Note that the auto-selection could fail if spin-glass",
                "SPO representation is requested.",
            ]
        ),
        default=-1,
    )
    parser.add_argument(
        "-use_spinglass_sparse_mpo",
        "--use_spinglass_sparse_mpo",
        help="If specified, enables efficient MPO representation for spin glasses.",
        action="store_true",
    )
    parser.add_argument(
        "-linalg_backend_library",
        "--linalg_backend_library",
        choices=["numpy", "torch"],
        help=" ".join(
            [
                "Linear algebra library to handle computations in qtealeaves",
                "at the tensor level.",
                "Available options are 'numpy' (and automatically 'cupy' when",
                "device is GPU) and 'torch' (both for CPU and GPU)",
                "Default to 'numpy'.",
            ]
        ),
        default="numpy",
    )
    parser.add_argument(
        "-device",
        "--device",
        choices=["cpu", "gpu"],
        help=" ".join(
            [
                "Computational device for tensor-network operations.",
                "Could be 'cpu' or 'gpu'. Default to 'cpu'.",
            ]
        ),
        default="cpu",
    )


def postprocess_cli_options(cli_options):
    """
    Post-processes the parsed command-line interface
    (CLI) options for this library simulations.

    Argguments
    ----------
    cli_options : Dict[str, Any]
        The parsed CLI options dictionary.

    Returns
    -------
    :obj:`None`

    Raises
    ------
    :exc:`ValueError`
        If required options are missing or invalid configurations
        are provided.
    """

    # Initialize constraints dictionary
    cli_options["constraints"] = {}

    for ii in range(1, 8):
        # Retrieve the flag for the constraint
        flag = cli_options.get(f"constraint_{ii}_flag", False)

        # Retrieve the penalty ratio if the constraint flag is True
        penalty_ratio = (
            cli_options.get(f"constraint_{ii}_qubo_penalty_constant_ratio")
            if flag
            else None
        )

        # Validate the penalty ratio (if provided)
        if penalty_ratio is not None and (
            not isinstance(penalty_ratio, (int, float)) or penalty_ratio <= 0
        ):
            raise ValueError(
                f"Invalid penalty constant ratio for constraint C{ii}. "
                f"Expected a positive float, but got: {penalty_ratio}"
            )

        # Add the constraint information to the dictionary
        cli_options["constraints"][f"C{ii}"] = (flag, penalty_ratio)

    # Validate the initialization mode and associated input file
    if cli_options.get("instance_initialization_mode") == "from_file":
        input_file = cli_options.get("instance_input_file")
        if not input_file:
            raise ValueError(
                "Initialization mode `from_file` requires an "
                "input file to read the MP4EO problem instance."
            )

    # Validate the tensor-network data-type during sweeping
    mode = cli_options.get("tensors_data_type_mode")
    dtype = cli_options.get("tensors_data_type")
    num_sweeps = cli_options.get("number_of_sweeps")

    if isinstance(dtype, list):
        if len(dtype) == 1:
            dtype = dtype[0]
        elif len(dtype) != num_sweeps:
            raise ValueError(
                "Tensors data-type is not specified for each sweep."
            )
        elif mode != "homogeneous":
            raise ValueError(
                "Already specified per-sweep data-type for tensors."
            )

    if mode in {"half-increasing", "half-decreasing"}:
        increasing_map = {"C": "Z", "S": "D"}
        decreasing_map = {"Z": "C", "D": "S"}

        if mode == "half-increasing":
            dtype_map = increasing_map
        else:
            dtype_map = decreasing_map

        if dtype not in dtype_map:
            raise ValueError(
                f"Invalid tensors data-type mode '{mode}' selected for the "
                "tensor-network solver: cannot "
                f"{'increase' if mode == 'half-increasing' else 'decrease'}"
                f" further than {dtype}."
            )

        half_sweeps = num_sweeps // 2
        dtype = [dtype] * half_sweeps + [dtype_map[dtype]] * (
            num_sweeps - half_sweeps
        )

    cli_options["tensors_data_type"] = dtype

##############################################################################
#                      Tensor Embedding Utilities                           #
##############################################################################
# cost function proj and number proj
def embed_one_body(proj_list, local_dims):
    """
    Embed the cost function (Z) as a one-body operator into the full Hilbert space.

    Parameters
    ----------
    proj_list : List[csr_matrix]
        List of cost function local one-qudit projectors to embed.

    local_dims : List[int]
        List of local Hilbert space dimensions for each physical qudit.

    Returns
    -------
    ops : List[csr_matrix]
        List of full-space operators obtained by embedding each one-body projector
        at the corresponding qudit position, surrounded by identity operators.
    """
    ops = []
    for site_idx, local_op in enumerate(proj_list): #for site_idx, local_op in proj_dict.items():
        # Left identity chain
        left = ssp.identity(1, format="csr")
        for i in range(site_idx):
            left = ssp.kron(left, ssp.identity(local_dims[i], format="csr"), format="csr")

        # Right identity chain
        right = ssp.identity(1, format="csr")
        for i in range(site_idx + 1, len(local_dims)):
            right = ssp.kron(right, ssp.identity(local_dims[i], format="csr"), format="csr")

        # Embed
        full_op = ssp.kron(ssp.kron(left, local_op, format="csr"), right, format="csr")

        assert full_op.shape == (np.prod(local_dims), np.prod(local_dims)), \
        "Cost function operator shape does not match the expected size"

        ops.append(full_op)

    return ops

# c3 and c4
def embed_two_body_same_site(proj_dict, local_dims):
    """
    Embed two-body operators acting on (DTO, DLO) pairs of the same mission site
    into the full Hilbert space.

    Parameters
    ----------
    proj_dict : Dict[Tuple[int, int], scipy.sparse.dia_matrix]
        Dictionary of two-qudit projectors acting on the same site, with site indices as key tuple and 
        matrix projcetor as values.

    local_dims : List[int]
        List of local Hilbert space dimensions for each physical qudit.

    Returns
    -------
    ops : List[csr_matrix]
        List of full-space operators obtained by embedding each two-qudit projector
        on the corresponding (DTO, DLO) pair, surrounded by identity operators.
    """
    ops = []

    for idx, op in proj_dict.items():
        # qudit physical indices
        dto_idx = 2 * idx
        dlo_idx = 2 * idx + 1

        # left Id
        left_chain = ssp.identity(1, format="csr")
        for i in range(dto_idx):
            left_chain = ssp.kron(left_chain, ssp.identity(local_dims[i], format="csr"), format="csr")

        # right Id
        right_chain = ssp.identity(1, format="csr")
        for i in range(dlo_idx + 1, len(local_dims)):
            right_chain = ssp.kron(right_chain, ssp.identity(local_dims[i], format="csr"), format="csr")

        # tensorize
        full_op = ssp.kron(
            ssp.kron(left_chain, op, format="csr"),
            right_chain,
            format="csr"
        )

        assert full_op.shape == (np.prod(local_dims), np.prod(local_dims)), \
        "Tensorized operator shape does not match the expected size (two-body-same-site operator)"
        ops.append(full_op)

    return ops

# c5 and c6
def embed_two_body(local_proj_j, local_proj_s, local_dims, qudit_type, site_idx_j, site_idx_s):
    """
    Tenosrize projectors acting on qudits located at different sites
    (DTO-DTO or DLO-DLO), into the full Hilbert space.

    This function takes two single-site projectors (diagonal matrix with just one element)
    acting on two distintic physical sites and tensorized them into the full Hilbert space.


    Parameters
    ----------
    local_proj_j : csr_matrix
        Diagonal matrix with rank-1 acting locally only on qudit at physical site `site_idx_j`. 

    local_proj_j : csr_matrix
        Diagonal matrix with rank-1 acting locally only on qudit at physical site `site_idx_s`. 
        
    local_dims : List[int]
        List of local Hilbert space dimensions for each qudit in the full system.

    qudit_type : str
        Either "DTO" or "DLO", indicating whether the involved qudits correspond to Data Take 
        Opportunities or Downlink Opportunities, respectively. This determines the mapping 
        from site indices to qudit indices. 

    site_idx_j : int
        Physical site index of the first qudit.

    site_idx_s : int
        Physical site index of the second qudit.

    Returns
    -------
    full_op : csr_matrix
        Sparse matrix representation of the full many-qudit operator, constructed by placing the 
        two projectors at their corresponding qudit positions and tensoring with identity 
        operators elsewhere.
    """

    get_qidx = (lambda j: 2*j) if qudit_type == "DTO" else (lambda j: 2*j+1)

    qj, qs = get_qidx(site_idx_j), get_qidx(site_idx_s)

    if qj <= qs:
        q1,q2 = qj,qs
        P1,P2 = local_proj_j, local_proj_s
    
    else:
        q1,q2 = qs, qj
        P1,P2 = local_proj_s, local_proj_j

    assert P1.shape[0] == local_dims[q1], f"P1 shape {P1.shape} ≠ dim[{q1}]={local_dims[q1]}"
    assert P2.shape[0] == local_dims[q2], f"P2 shape {P2.shape} ≠ dim[{q2}]={local_dims[q2]}"


    # left Id
    left_chain = ssp.identity(1, format="csr")
    for i in range(q1):
        left_chain = ssp.kron(left_chain, ssp.identity(local_dims[i], format="csr"), format="csr")

    # mid Id
    mid_chain = ssp.identity(1, format = "csr")
    for i in range(q1+1, q2):
        mid_chain = ssp.kron(mid_chain, ssp.identity(local_dims[i], format="csr"), format="csr")
    
    # right Id
    right_chain = ssp.identity(1, format="csr")
    for i in range(q2+1, len(local_dims)):
        right_chain = ssp.kron(right_chain, ssp.identity(local_dims[i], format="csr"), format="csr")

    full_op = ssp.kron(ssp.kron(ssp.kron(ssp.kron(left_chain, P1, format="csr"), mid_chain, format="csr"), P2, format="csr"), right_chain, format="csr")
    
    assert full_op.shape == (np.prod(local_dims), np.prod(local_dims)), \
    "Tensorized operator shape does not match the expected size (two-body operator)"


    return full_op
