import pytest
import numpy as np
import netket as nk
import bridge

from test import common


@common.xfailif_distributed
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_qubits, m_states", [(5, 4), (3, 2)])
def test_observable_estimation(n_qubits, m_states, seed):
    """The current threshold is not very ambitious."""

    hilbert_space = nk.hilbert.Qubit(n_qubits)

    graph = nk.graph.Chain(n_qubits, pbc=True)
    hamiltonian = nk.operator.IsingJax(
        hilbert_space, graph, 1, 1, dtype=complex
    ).to_local_operator()

    model = nk.models.RBM(alpha=2)
    states = [nk.vqs.FullSumState(hilbert_space, model) for _ in range(m_states)]

    coefficients = np.sin(np.linspace(0, 1000, m_states))
    linear_combination_array = sum(
        coefficients[k] * state.to_array(normalize=False)
        for k, state in enumerate(states)
    )

    energy_exact = (
        np.einsum(
            "i,ij,j",
            linear_combination_array.conj(),
            hamiltonian.to_dense(),
            linear_combination_array,
        )
        / np.linalg.norm(linear_combination_array) ** 2
    )

    linear_state = bridge.construct_linear_state(states, coefficients)

    assert abs(energy_exact - linear_state.expect(hamiltonian).mean).item() < 1e-14
    assert (
        np.linalg.norm(
            linear_state.to_array(normalize=False) - linear_combination_array
        )
        / linear_combination_array.size
        < 1e-15
    )

    sampling_rule = nk.sampler.rules.LocalRule()
    gram_matrix, projected_hamiltonian = (
        bridge.estimate_projected_operators_sum_of_states(
            states,
            [hamiltonian],
            n_samples=300000,
            sampling_rule=sampling_rule,
            seed=seed,
        )
    )

    numerator = np.einsum(
        "i,ij,j", coefficients.conj(), projected_hamiltonian, coefficients
    )
    denominator = np.einsum("i,ij,j", coefficients.conj(), gram_matrix, coefficients)
    estimated_energy = numerator / denominator

    assert abs((estimated_energy - energy_exact) / energy_exact) < 3e-3
