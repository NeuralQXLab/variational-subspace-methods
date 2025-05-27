"""Test the estimation of the Rayleigh matrix of a full basis.
Because of the zero-variance property of the determinant estimator, the results should be good irrespective of the number of samples.
The Hamiltonian used has degenerate eigenvalues, so it is complicated to test the eigenvectors.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import bridge


@pytest.mark.parametrize("n_samples", [1, 30, 3000])
def test_full_rayleigh_matrix_rbm(n_samples):
    """The RBMs used in this test initialize very close to each other, such that the number of qubits must be small and the required thresholds must large."""
    if jax.process_count() > 1:
        pytest.xfail("not working for multi-process.")

    n_qubits = 2
    rng_list = [jax.random.PRNGKey(k) for k in range(2**n_qubits)]
    hilbert_space = nk.hilbert.Qubit(n_qubits)
    graph = nk.graph.Chain(n_qubits, pbc=False)
    hamiltonian = nk.operator.IsingJax(hilbert_space, graph, 1, 1).to_local_operator()

    model = nk.models.RBM(param_dtype=complex)
    states = [
        nk.vqs.FullSumState(hilbert_space, model, seed=rng_list[k])
        for k in range(2**n_qubits)
    ]
    states_np = np.array([state.to_array(normalize=False) for state in states])

    gram_matrix = states_np.conj() @ states_np.T
    projected_hamiltonian = states_np.conj() @ hamiltonian.to_dense() @ states_np.T
    rm_exact = np.linalg.solve(gram_matrix, projected_hamiltonian)

    sampling_rule = nk.sampler.rules.LocalRule()
    rme = bridge.estimate_rayleigh_matrix_determinant_state(
        states,
        hamiltonian,
        n_samples=n_samples,
        n_chains=1,
        sampling_rule=sampling_rule,
    )

    rm_error = np.linalg.norm(np.abs(rm_exact - rme) / rm_exact)

    eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_dense())
    eigvecs = eigvecs.T

    rme_eigvals, rme_eigvecs = np.linalg.eig(rme)
    rme_ordering = np.argsort(rme_eigvals)
    rme_eigvals = rme_eigvals[rme_ordering]
    # rme_eigvecs = (rme_eigvecs.T @ states_np)[rme_ordering]

    # infidelity = 1 - bridge.numpy_tools.fidelity(eigvecs, rme_eigvecs)
    # error_eigvecs = np.linalg.norm(infidelity**0.5) / 2**n_qubits
    error_eigvals = np.linalg.norm(eigvals - rme_eigvals) / 2**n_qubits

    assert error_eigvals < 1e-5
    # assert error_eigvecs < 1e-5
    assert rm_error < 1e-5


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [1, 30, 3000])
def test_full_rayleigh_matrix_logstatevec(n_qubits, n_samples):
    if jax.process_count() > 1:
        pytest.xfail("not working for multi-process.")

    hilbert_space = nk.hilbert.Qubit(n_qubits)
    graph = nk.graph.Chain(n_qubits, pbc=False)
    hamiltonian = nk.operator.IsingJax(hilbert_space, graph, 1, 1).to_local_operator()

    model = nk.models.LogStateVector(hilbert_space)
    states = [nk.vqs.FullSumState(hilbert_space, model) for _ in range(2**n_qubits)]
    for state in states:
        state.variables = {
            "params": {
                "logstate": jnp.array(
                    np.random.uniform(-1, 1, 2**n_qubits)
                    + 1.0j * np.random.uniform(-1, 1, 2**n_qubits)
                )
            }
        }
    states_np = np.array([state.to_array(normalize=False) for state in states])

    gram_matrix = states_np.conj() @ states_np.T
    projected_hamiltonian = states_np.conj() @ hamiltonian.to_dense() @ states_np.T
    rm_exact = np.linalg.solve(gram_matrix, projected_hamiltonian)

    sampling_rule = nk.sampler.rules.LocalRule()
    rme = bridge.estimate_rayleigh_matrix_determinant_state(
        states,
        hamiltonian,
        n_samples=n_samples,
        n_chains=1,
        sampling_rule=sampling_rule,
    )

    rm_error = np.linalg.norm(np.abs(rm_exact - rme) / rm_exact)

    eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_dense())
    eigvecs = eigvecs.T

    rme_eigvals = np.linalg.eig(rme).eigenvalues
    rme_eigvals = np.sort(rme_eigvals)
    error_eigvals = np.linalg.norm(eigvals - rme_eigvals) / 2**n_qubits

    assert error_eigvals < 1e-10
    assert rm_error < 1e-10
