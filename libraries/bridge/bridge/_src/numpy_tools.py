import numpy as np


def reduce_operator(operator, states):
    """Compute the reduced operator in the subspace spanned by the given states. The array states should be of shape (n_states, state_dim)."""
    return np.conj(states) @ (operator @ states.T)


def fidelity(array_psi, array_phi):
    """Compute the fidelities between 2d arrays representing lists of states. The inputs have dimension (n_states, state_dim)."""
    assert array_psi.shape == array_phi.shape
    overlaps = np.einsum("mn,mn->m", np.conj(array_psi), array_phi)
    psi_norms_squared = np.einsum("mn,mn->m", np.conj(array_psi), array_psi)
    phi_norms_squared = np.einsum("mn,mn->m", np.conj(array_phi), array_phi)
    return np.abs(overlaps) ** 2 / (psi_norms_squared * phi_norms_squared)
