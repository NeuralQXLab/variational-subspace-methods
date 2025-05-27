import pytest
import numpy as np

import jax
import jax.numpy as jnp
import netket as nk
import bridge

import qutip as qt

# Inverting Gram in high precision is cursed so we don't do it.
# For some reason, determinant state is better than optimal (no unexpected),
# but then solve Gram is very close to optimal, and pinv with the best rcond is about 1 or 2 orders of magnitude worse.
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
        "gram_rcond": 1e-14,
        "gram_decimal_precision": None,
    },
]


@pytest.mark.parametrize(
    "n_qubits, m_states, threshold_half, threshold_full",
    [(5, 10, 1e-6, 1e-4), (6, 14, 1e-10, 5e-4)],
)
@pytest.mark.parametrize("chunk_size", [None, 1])
@pytest.mark.parametrize("discard_imaginary", [False])
@pytest.mark.parametrize("decimal_precision_solver", [None, 100])
@pytest.mark.parametrize("additional_bridge_args", additional_bridge_args_list)
def test_discretized_basis(
    n_qubits,
    m_states,
    threshold_half,
    threshold_full,
    chunk_size,
    discard_imaginary,
    decimal_precision_solver,
    additional_bridge_args,
):
    """Test bridge when the basis contains the second order discretize time evolution states."""

    if jax.process_count() > 1:
        pytest.xfail("not working for multi-process.")

    times = np.linspace(0, 1, m_states)

    hilbert_space = nk.hilbert.Qubit(n_qubits)
    graph = nk.graph.Chain(n_qubits, pbc=False)
    hamiltonian = nk.operator.IsingJax(
        hilbert_space, graph, 1, 1, dtype=complex
    ).to_local_operator()
    random_coefficients = np.sin(np.linspace(0, 1000, m_states))
    hamiltonian += sum(
        random_coefficients[k] * nk.operator.spin.sigmay(hilbert_space, k)
        for k in range(n_qubits)
    )

    sampler = nk.sampler.MetropolisLocal(hilbert_space)
    model = nk.models.LogStateVector(hilbert_space)
    states = [nk.vqs.MCState(sampler, model) for _ in range(m_states)]

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

    # Discretized dynamics
    delta = times[1]
    discretized_dynamics_states = [psi_0]
    for _ in range(m_states - 1):
        current_state = discretized_dynamics_states[-1]
        discretized_dynamics_states.append(
            current_state
            - 1.0j * delta * hamiltonian_qt * current_state
            - 0.5 * delta**2 * hamiltonian_qt**2 * psi_0
        )
    discretized_dynamics_states_np = np.array(
        [state.full()[:, 0] for state in discretized_dynamics_states]
    )

    # The basis states for bridge are the discretized dynamics states
    for k, state in enumerate(states):
        state.variables = {
            "params": {
                "logstate": jnp.array(jnp.log(discretized_dynamics_states_np[k]))
            }
        }
    states_np = np.array([state.to_array(normalize=False) for state in states])

    # Bridge
    bridge_states, rayleigh_matrix_estimate, info_dict = bridge.bridge(
        states,
        hamiltonian,
        times,
        chunk_size=chunk_size,
        decimal_precision_solver=decimal_precision_solver,
        discard_imaginary=discard_imaginary,
        **additional_bridge_args,
        timeit=False,
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

    if (
        "gram_rcond" in additional_bridge_args
        and additional_bridge_args["gram_rcond"] is not None
    ):
        threshold_half *= 100
        threshold_full *= 100

    assert (
        np.linalg.norm(optimal_infidelity[: m_states // 2])
        / optimal_infidelity[: m_states // 2].size
        < threshold_half
    )
    assert np.linalg.norm(optimal_infidelity) / optimal_infidelity.size < threshold_full

    assert (
        np.linalg.norm(bridge_states_infidelity[: m_states // 2])
        / bridge_states_infidelity[: m_states // 2].size
        < threshold_half
    )
    assert (
        np.linalg.norm(bridge_states_infidelity) / bridge_states_infidelity.size
        < threshold_full
    )
