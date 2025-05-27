import pytest
import numpy as np
import netket as nk
import bridge


@pytest.mark.parametrize("m_states", [10, 20])
@pytest.mark.parametrize("cls", [nk.vqs.MCState, nk.vqs.FullSumState, None])
@pytest.mark.parametrize("n_samples", [None, 512])
@pytest.mark.parametrize("chunk_size", [None, 4032])
def test_linear_combination_from_fullsum(m_states, cls, n_samples, chunk_size):

    n_qubits = 5
    hilbert_space = nk.hilbert.Qubit(n_qubits)
    sampler = nk.sampler.MetropolisLocal(hilbert_space)

    coefficients = np.sin(np.linspace(0, 1000, m_states))
    model = nk.models.RBM(alpha=2)

    states = [nk.vqs.FullSumState(hilbert_space, model) for _ in range(m_states)]

    linear_combination_array = sum(
        coefficients[k] * state.to_array(normalize=False)
        for k, state in enumerate(states)
    )

    linear_state = bridge.construct_linear_state(
        states,
        coefficients,
        cls=cls,
        seed=0,
        sampler=sampler,
        n_samples=n_samples,
        chunk_size=chunk_size,
    )

    assert (
        np.linalg.norm(
            linear_state.to_array(normalize=False) - linear_combination_array
        )
        / linear_combination_array.size
        < 1e-15
    )

    assert linear_state.chunk_size == chunk_size

    if cls == nk.vqs.MCState:

        assert isinstance(linear_state, nk.vqs.MCState)

        if n_samples is not None:
            assert linear_state.n_samples == n_samples
            

    if cls == nk.vqs.FullSumState or cls is None:
        assert isinstance(linear_state, nk.vqs.FullSumState)


@pytest.mark.parametrize("m_states", [10, 20])
@pytest.mark.parametrize("cls", [nk.vqs.MCState, nk.vqs.FullSumState, None])
@pytest.mark.parametrize("n_samples", [None, 512])
@pytest.mark.parametrize("chunk_size", [None, 16])
def test_linear_combination_from_mcstate(m_states, cls, n_samples, chunk_size):

    n_qubits = 5
    hilbert_space = nk.hilbert.Qubit(n_qubits)
    sampler = nk.sampler.MetropolisLocal(hilbert_space)

    coefficients = np.sin(np.linspace(0, 1000, m_states))
    model = nk.models.RBM(alpha=2)

    old_n_samples = 256
    old_chunk_size = 8

    states = [
        nk.vqs.MCState(
            sampler, model, n_samples=old_n_samples, chunk_size=old_chunk_size
        )
        for _ in range(m_states)
    ]

    linear_combination_array = sum(
        coefficients[k] * state.to_array(normalize=False)
        for k, state in enumerate(states)
    )

    linear_state = bridge.construct_linear_state(
        states,
        coefficients,
        cls=cls,
        seed=0,
        sampler=sampler,
        n_samples=n_samples,
        chunk_size=chunk_size,
    )

    assert (
        np.linalg.norm(
            linear_state.to_array(normalize=False) - linear_combination_array
        )
        / linear_combination_array.size
        < 1e-15
    )

    if chunk_size is None:
        assert linear_state.chunk_size == old_chunk_size
    else:
        assert linear_state.chunk_size == chunk_size

    if cls == nk.vqs.MCState or cls is None:

        assert isinstance(linear_state, nk.vqs.MCState)

        if n_samples is None:
            assert linear_state.n_samples == states[0].n_samples
        else:
            assert linear_state.n_samples == n_samples

    if cls == nk.vqs.FullSumState:
        assert isinstance(linear_state, nk.vqs.FullSumState)
