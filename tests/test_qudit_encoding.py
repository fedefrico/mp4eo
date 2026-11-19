"""Synthetic, in-memory checks for the constraint-aware qudit basis.

The small instance below is specified directly in this file. It is not sampled
from, or calibrated against, any industrial dataset. C7 is deliberately out of
scope; the checks cover the C1/C2 basis and C3-C6 exact projectors.
"""

from itertools import product
from math import prod

import numpy as np
import pytest

from mp4eo_optimization.mp4eo_instance import MP4EOInstance
from mp4eo_optimization.mp4eo_mappers import QuditsMapper


@pytest.fixture
def synthetic_instance(tmp_path, monkeypatch):
    """Build a tiny deterministic instance without loading external data."""
    monkeypatch.chdir(tmp_path)

    n_satellites = 2
    n_targets = 2
    theta = np.array([1, 2, 2, 1], dtype=int)
    sigma = np.array([1, 2], dtype=int)
    n_dto = int(theta.sum())
    n_dlo = n_targets * int(sigma.sum())

    constraints = {
        f"C{index}": (index in range(1, 7), 1.0)
        for index in range(1, 8)
    }

    return MP4EOInstance(
        n_satellites=n_satellites,
        n_targets=n_targets,
        theta=theta,
        n_dto=n_dto,
        dto_times=np.array([10, 14, 11, 17, 13, 18], dtype=float),
        dto_preparation_times=np.full(n_dto, 2.5, dtype=float),
        sigma=sigma,
        n_dlo=n_dlo,
        dlo_times=np.array([13, 12, 18, 19, 16, 17], dtype=float),
        dlo_preparation_times=np.full(n_dlo, 2.5, dtype=float),
        required_target_memories=np.array([1, 2, 2, 1], dtype=float),
        total_available_memories=np.array([8, 8], dtype=float),
        alpha=1.0,
        beta=1.0,
        gamma=0.0,
        omega_dto=np.array([1, 3, 2, 5, 4, 6], dtype=float),
        omega_dlo=np.array([2, 4, 3, 5, 4, 6], dtype=float),
        iparam_range=None,
        binaries_arrangement="target-wise",
        constraints=constraints,
    )


def test_qudit_decoder_bijects_onto_c1_c2_feasible_space(synthetic_instance):
    """Each local occupation decodes to one distinct C1/C2-feasible mission."""
    instance = synthetic_instance
    mapper = QuditsMapper(instance, build_mode="exact")

    local_dimensions = []
    for target in range(instance.n_targets):
        local_dimensions.extend(
            [
                len(mapper.get_jth_dto_logical_states(target)),
                len(mapper.get_jth_dlo_logical_states(target)),
            ]
        )

    decoded_missions = set()
    for occupations in product(*(range(dim) for dim in local_dimensions)):
        mission = mapper.qudit_occupations_to_mission_bitstring(occupations)
        mission = tuple(int(bit) for bit in mission)

        assert instance.check_constraint_1(mission) == (
            instance.n_targets,
            instance.n_targets,
        )
        assert instance.check_constraint_2(mission) == (
            instance.n_targets,
            instance.n_targets,
        )
        decoded_missions.add(mission)

    # The number of C1/C2-feasible assignments is the product of the local
    # DTO and DLO choices for each target. Distinct decoded states therefore
    # establish that the mapper covers that whole subspace exactly once.
    assert len(decoded_missions) == prod(local_dimensions)


def test_qudit_constraint_projectors_match_direct_c3_to_c6_checks(
    synthetic_instance,
):
    """Exact projector diagonals agree with direct checks on decoded states."""
    instance = synthetic_instance
    mapper = QuditsMapper(instance, build_mode="exact")
    penalty = 7.0
    mapper.map(penalty=penalty, build_mode="exact")

    local_dimensions = mapper.local_dimensions
    c5_diagonal = mapper.C5_projectors.diagonal()
    c6_diagonal = mapper.C6_projectors.diagonal()
    saw_c3_violation = False
    saw_c4_violation = False
    saw_c5_violation = False
    saw_c6_violation = False

    for occupations in product(*(range(dim) for dim in local_dimensions)):
        mission = mapper.qudit_occupations_to_mission_bitstring(occupations)
        mission = tuple(int(bit) for bit in mission)
        c3_satisfied, c3_total = instance.check_constraint_3(mission)
        c4_satisfied, c4_total = instance.check_constraint_4(mission)
        c5_satisfied, c5_total = instance.check_constraint_5(mission)
        c6_satisfied, c6_total = instance.check_constraint_6(mission)

        c3_energy = 0.0
        c4_energy = 0.0
        for target in range(instance.n_targets):
            dto_level = occupations[2 * target]
            dlo_level = occupations[2 * target + 1]
            local_index = (
                dto_level * mapper.dlo_local_dimensions[target] + dlo_level
            )
            c3_energy += mapper.C3_projectors[target].diagonal()[local_index]
            c4_energy += mapper.C4_projectors[target].diagonal()[local_index]

        full_index = np.ravel_multi_index(occupations, local_dimensions)
        c5_energy = c5_diagonal[full_index]
        c6_energy = c6_diagonal[full_index]

        # A different DTO and DLO satellite creates two failed C3 equalities
        # per target, while the local projector assigns one penalty per target.
        assert (c3_total - c3_satisfied) % 2 == 0
        expected_c3_energy = (
            (c3_total - c3_satisfied) // 2
        ) * (penalty + 0.1)
        assert c3_energy == pytest.approx(expected_c3_energy)
        saw_c3_violation |= c3_satisfied < c3_total

        # C4's projector encodes same-satellite timing conflicts. Restrict
        # this comparison to C3-feasible states so unpaired acquisition or
        # downlink choices do not contribute C4 failures in the direct check.
        if c3_satisfied == c3_total:
            c4_violations = c4_total - c4_satisfied
            assert c4_energy == pytest.approx(c4_violations * (penalty + 0.1))
            saw_c4_violation |= c4_violations > 0

        c5_violations = c5_total - c5_satisfied
        c6_violations = c6_total - c6_satisfied
        assert c5_energy == pytest.approx(c5_violations * penalty)
        assert c6_energy == pytest.approx(c6_violations * penalty)
        saw_c5_violation |= c5_violations > 0
        saw_c6_violation |= c6_violations > 0

    assert saw_c3_violation
    assert saw_c4_violation
    assert saw_c5_violation
    assert saw_c6_violation
