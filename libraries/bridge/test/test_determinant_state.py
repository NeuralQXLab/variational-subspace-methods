import pytest
import numpy as np
import netket as nk
import bridge

from itertools import permutations



def get_sign(permutation):
    larger_indices, smaller_indices = np.tril_indices(permutation.size, k=-1)
    sign_table = np.where(permutation[larger_indices] > permutation[smaller_indices], 1, -1)
    return np.prod(sign_table)

def tensor_permutation(states, permutation):
    res = states[permutation[0]]
    for k in range(1, permutation.size):
        res = np.kron(res, states[permutation[k]])
    return res



@pytest.mark.parametrize("n_qubits, m_states", [(2, 3), (4, 2)])
@pytest.mark.parametrize("model", [nk.models.RBM(alpha=2)])
@pytest.mark.parametrize("cls", [nk.vqs.MCState, nk.vqs.FullSumState, None])
@pytest.mark.parametrize("n_samples", [None, 512])
@pytest.mark.parametrize("sweep_size", [None, 20])
@pytest.mark.parametrize("n_chains", [None, 512])
@pytest.mark.parametrize("n_discard_per_chain", [None, 15])
@pytest.mark.parametrize("chunk_size", [None, 16])
def test_determinant_from_fullsum(n_qubits, m_states, model, cls, n_samples, sweep_size, n_chains, n_discard_per_chain, chunk_size):

    hilbert_space = nk.hilbert.Qubit(n_qubits)
    
    permutation_array = np.array(list(permutations(range(m_states), m_states)))
    sign_array = np.array([get_sign(permutation) for permutation in permutation_array])

    states = [nk.vqs.FullSumState(hilbert_space, model) for _ in range(m_states)]

    states_np = np.array([state.to_array(normalize=False) for state in states])
    permuted_tensor_products = np.array([tensor_permutation(states_np, permutation) for permutation in permutation_array])
    det_state_1 = np.sum(sign_array[:, np.newaxis] * permuted_tensor_products, axis=0)

    det_vstate = bridge.construct_determinant_state(states, cls=cls, n_samples=n_samples, sweep_size=sweep_size, n_chains=n_chains, n_discard_per_chain=n_discard_per_chain, chunk_size=chunk_size, sampling_rule=nk.sampler.rules.LocalRule(), seed=0)
    det_state_2 = det_vstate.to_array(normalize=False)

    assert (np.linalg.norm(det_state_1 - det_state_2) / det_state_1.size < 1e-15)

    assert det_vstate.chunk_size == chunk_size

    if cls == nk.vqs.MCState:

        assert isinstance(det_vstate, nk.vqs.MCState)

        if n_samples is not None:
            assert det_vstate.n_samples == n_samples
        if sweep_size is not None:
            assert det_vstate.sampler.sweep_size == sweep_size
        if n_chains is not None:
            assert det_vstate.sampler.n_chains == n_chains
        if n_discard_per_chain is not None:
            assert det_vstate.n_discard_per_chain == n_discard_per_chain

    if cls == nk.vqs.FullSumState or cls is None:
        assert isinstance(det_vstate, nk.vqs.FullSumState)




@pytest.mark.parametrize("n_qubits, m_states", [(2, 3), (4, 2)])
@pytest.mark.parametrize("model", [nk.models.RBM(alpha=2)])
@pytest.mark.parametrize("cls", [nk.vqs.MCState, nk.vqs.FullSumState, None])
@pytest.mark.parametrize("n_samples", [None, 512])
@pytest.mark.parametrize("sweep_size", [None, 20])
@pytest.mark.parametrize("n_chains", [None, 512])
@pytest.mark.parametrize("n_discard_per_chain", [None, 15])
@pytest.mark.parametrize("chunk_size", [None, 16])
def test_determinant_from_mcstate(n_qubits, m_states, model, cls, n_samples, sweep_size, n_chains, n_discard_per_chain, chunk_size):

    hilbert_space = nk.hilbert.Qubit(n_qubits)
    
    old_chunk_size = 8
    sampler = nk.sampler.MetropolisLocal(hilbert_space)
    
    permutation_array = np.array(list(permutations(range(m_states), m_states)))
    sign_array = np.array([get_sign(permutation) for permutation in permutation_array])

    states = [nk.vqs.MCState(sampler, model, chunk_size=old_chunk_size) for _ in range(m_states)]

    states_np = np.array([state.to_array(normalize=False) for state in states])
    permuted_tensor_products = np.array([tensor_permutation(states_np, permutation) for permutation in permutation_array])
    det_state_1 = np.sum(sign_array[:, np.newaxis] * permuted_tensor_products, axis=0)

    det_vstate = bridge.construct_determinant_state(states, cls=cls, n_samples=n_samples, sweep_size=sweep_size, n_chains=n_chains, n_discard_per_chain=n_discard_per_chain, chunk_size=chunk_size, seed=0)
    det_state_2 = det_vstate.to_array(normalize=False)

    assert (np.linalg.norm(det_state_1 - det_state_2) / det_state_1.size < 1e-15)

    if chunk_size is None:
        assert det_vstate.chunk_size == old_chunk_size
    else:
        assert det_vstate.chunk_size == chunk_size

    if cls == nk.vqs.MCState or cls is None:

        assert isinstance(det_vstate, nk.vqs.MCState)

        if n_samples is not None:
            assert det_vstate.n_samples == n_samples
        if sweep_size is not None:
            assert det_vstate.sampler.sweep_size == sweep_size
        if n_chains is not None:
            assert det_vstate.sampler.n_chains == n_chains
        if n_discard_per_chain is not None:
            assert det_vstate.n_discard_per_chain == n_discard_per_chain

    if cls == nk.vqs.FullSumState:
        assert isinstance(det_vstate, nk.vqs.FullSumState)