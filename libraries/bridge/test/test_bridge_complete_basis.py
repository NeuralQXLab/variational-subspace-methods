import pytest
import numpy as np
import jax.numpy as jnp
import netket as nk
import bridge

import qutip as qt

additional_bridge_args_list = [
    {"n_samples": 3000, "estimation_method": "determinant_state"},
    {
        "n_samples": 30000,
        "estimation_method": "sum_of_states",
        "gram_rcond": None,
        "gram_decimal_precision": None,
    },
    {
        "n_samples": 30000,
        "estimation_method": "sum_of_states",
        "gram_rcond": 1e-12,
        "gram_decimal_precision": None,
    },
    {
        "n_samples": 30000,
        "estimation_method": "sum_of_states",
        "gram_rcond": None,
        "gram_decimal_precision": 100,
    },
]


@pytest.mark.parametrize("n_qubits, m_states", [(5, 5), (6, 7)])
@pytest.mark.parametrize("chunk_size", [None, 1])
@pytest.mark.parametrize("discard_imaginary", [True, False])
@pytest.mark.parametrize("decimal_precision_solver", [None, 100])
@pytest.mark.parametrize("additional_bridge_args", additional_bridge_args_list)
def test_complete_basis(
    n_qubits,
    m_states,
    chunk_size,
    discard_imaginary,
    decimal_precision_solver,
    additional_bridge_args,
):
    """Test bridge when the basis contains the exact dynamics states."""

    times = np.linspace(0, 1, m_states)

    hilbert_space = nk.hilbert.Qubit(n_qubits)
    graph = nk.graph.Chain(n_qubits, pbc=False)
    hamiltonian = nk.operator.IsingJax(hilbert_space, graph, 1, 1).to_local_operator()

    sampling_rule = nk.sampler.rules.LocalRule()
    model = nk.models.LogStateVector(hilbert_space)
    states = [nk.vqs.FullSumState(hilbert_space, model) for _ in range(m_states)]

    # Exact dynamics
    psi_0 = qt.Qobj(np.ones(2**n_qubits), dims=[n_qubits * [2], n_qubits * [1]])
    psi_0 /= psi_0.norm()
    hamiltonian_qt = hamiltonian.to_qobj()
    options = {"atol": 1e-8, "rtol": 1e-8}
    exact_dynamics_states = qt.sesolve(
        hamiltonian_qt, psi_0, times, options=options
    ).states
    exact_dynamics_states_np = np.array(
        [state.full()[:, 0] for state in exact_dynamics_states]
    )

    # The basis states for bridge are the exact dynamics states
    for k, state in enumerate(states):
        state.variables = {
            "params": {"logstate": jnp.array(jnp.log(exact_dynamics_states_np[k]))}
        }
    states_np = np.array([state.to_array(normalize=False) for state in states])

    # Bridge
    bridge_states, rayleigh_matrix_estimate, info_dict = bridge.bridge(
        states,
        hamiltonian,
        times,
        sweep_size=50,
        sampling_rule=sampling_rule,
        chunk_size=chunk_size,
        decimal_precision_solver=decimal_precision_solver,
        discard_imaginary=discard_imaginary,
        **additional_bridge_args,
        timeit=False,
    )

    # Infidelity of the basis states
    base_states_infidelities = np.real(
        1 - bridge.numpy_tools.fidelity(states_np, exact_dynamics_states_np)
    )

    # Infidelity of the bridge states
    bridge_states_full = np.inner(bridge_states, states_np.T)
    bridge_states_infidelity = np.real(
        1 - bridge.numpy_tools.fidelity(exact_dynamics_states_np, bridge_states_full)
    )

    # Optimal infidelity that can be obtained within the subspace spanned by the basis states
    optimal_infidelity = bridge.distance_to_subspace(
        exact_dynamics_states_np, states_np, decimal_precision=100
    )

    assert np.linalg.norm(optimal_infidelity) / optimal_infidelity.size < 1e-14
    assert (
        np.linalg.norm(base_states_infidelities) / base_states_infidelities.size < 1e-14
    )

    threshold = 5e-4 if discard_imaginary else 1e-6
    assert (
        np.linalg.norm(bridge_states_infidelity) / bridge_states_infidelity.size
        < threshold
    )

    # Exact Rayleigh matrix
    gram_matrix = states_np.conj() @ states_np.T
    projected_hamiltonian = states_np.conj() @ hamiltonian.to_dense() @ states_np.T
    rm_exact = np.linalg.solve(gram_matrix, projected_hamiltonian)

    assert np.linalg.norm(rm_exact - rayleigh_matrix_estimate) / rm_exact.size < 0.05
