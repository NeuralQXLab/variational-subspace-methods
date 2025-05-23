import numpy as np
import jax
import jax.numpy as jnp
import flax
import netket as nk
import mpmath as mp

from netket.vqs import VariationalState
from netket.sampler import MetropolisSampler

from bridge._src.models import LinearModel, NothingNet
from bridge._src.sum_of_states import ProbabilitySumState, estimate_everything


def concatenate_variables(*variables):
    """Concatenate JAX arrays along a new axis."""
    return jnp.concatenate(
        tuple(variable[jnp.newaxis, :] for variable in variables), axis=0
    )


def construct_linear_state(
    states: list[VariationalState],
    coefficients: np.ndarray,
    cls: type | None = None,
    seed: int | None = None,
    sampler: MetropolisSampler = None,
    n_samples: int | None = None,
    n_discard_per_chain: int | None = None,
    chunk_size: int | None = None,
):
    """
    Construct the variational state with a given basis of states and coefficients.

    Args:
        states (<MCState> or <FullSumState>): The states forming the basis.
        coefficents (ndarray): Coefficients of the linear combination.
        cls (type): Either nk.vqs.MCState or nk.vqs.FullSumState. If None, the determinant state is of the type of the first component state.
        seed (int or rng): If specified, the seed for the linear state.
        The remaining arguments specify properties of the variational states. If None, the corresponding property is taken from the first basis state if possible.
            - sampler
            - n_samples
            - n_discard_per_chain
            - chunk_size

    Return:
        MCState or FullSumState: The linear state.
    """

    m_states = len(states)
    hilbert_space = states[0].hilbert

    linear_model = LinearModel(
        base_network=NothingNet,
        base_arguments=flax.core.freeze({"base_net": states[0].model}),
        variable_keys=tuple(states[0].variables.keys()),
        m_states=m_states,
    )

    if cls is None:
        cls = type(states[0])
    
    if chunk_size is None:
        chunk_size = states[0].chunk_size

    if issubclass(cls, nk.vqs.FullSumState):

        linear_state = nk.vqs.FullSumState(hilbert_space, linear_model, chunk_size=chunk_size, seed=seed)

    elif issubclass(cls, nk.vqs.MCState):

        if sampler is None:
            if isinstance(states[0], nk.vqs.MCState):
                sampler = states[0].sampler
            else:
                raise ValueError(
                    "If the basis states are not MCState, then a sampling rule must be provided."
                )

        if n_samples is None and isinstance(states[0], nk.vqs.MCState):
            n_samples = states[0].n_samples

        if n_discard_per_chain is None and isinstance(states[0], nk.vqs.MCState):
            n_discard_per_chain = states[0].n_discard_per_chain

        linear_state = nk.vqs.MCState(
            sampler,
            linear_model,
            seed=seed,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            chunk_size=chunk_size,
        )

    else:
        raise TypeError(
            "The class can only be a subclass of either nk.vqs.FullSumState or nk.vqs.MCState."
        )

    # Construct the variables for the variational state
    concatenated_variables = jax.tree.map(
        concatenate_variables, *tuple(state.variables for state in states)
    )
    new_variables = {
        variable_name: {
            "base_arguments_base_net": concatenated_variables[variable_name]
        }
        for variable_name in concatenated_variables
    }
    new_variables["params"]["coefficients"] = coefficients

    linear_state.variables = new_variables

    return linear_state


def distance_to_subspace(states, subspace, decimal_precision=None):
    """
    For each state in states, compute the minimum infidelity to a given subspace described by its basis.

    Args:
        states (ndarray): The states for which we want the minimum infidelity to the subspace. Has shape (n_states, state_dim).
        subspace (ndarray): The basis states describing the subspace. Has shape (subspace_dim, state_dim).
        decimal_precision (int): If not None, uses mpmath with the specified decimal precision.

    Return:
        ndarray: Minimum infidelity for each state. Has shape (n_states,).
    """

    subspace_columns = subspace.T
    states_columns = states.T

    if decimal_precision is None:
        return _distance_to_subspace_np(states_columns, subspace_columns)
    else:
        return _distance_to_subspace_mp(
            states_columns, subspace_columns, decimal_precision
        )


def _distance_to_subspace_np(states_columns, subspace_columns):

    gram_matrix = np.conj(subspace_columns).T @ subspace_columns
    overlap_matrix = np.conj(subspace_columns).T @ states_columns
    linear_solution = np.linalg.solve(gram_matrix, overlap_matrix)
    projected_norms_square = np.real(
        np.diag(np.conj(overlap_matrix).T @ linear_solution)
    )
    norms_square = np.linalg.norm(states_columns, axis=0) ** 2

    return 1 - projected_norms_square / norms_square


def _distance_to_subspace_mp(states_columns, subspace_columns, decimal_precision):

    n_states = states_columns.shape[1]
    norms_square = np.linalg.norm(states_columns, axis=0) ** 2

    gram_matrix = subspace_columns.T.conjugate() @ subspace_columns
    overlap_matrix = subspace_columns.T.conjugate() @ states_columns

    with mp.workdps(decimal_precision):

        gram_matrix = mp.matrix(gram_matrix)
        overlap_matrix = mp.matrix(overlap_matrix)

        projected_norms_square_matrix = (
            overlap_matrix.T.conjugate() @ gram_matrix**-1 @ overlap_matrix
        ).apply(mp.re)

    projected_norms_square = np.diag(
        np.array(projected_norms_square_matrix, dtype=float).reshape(n_states, n_states)
    )

    return 1 - projected_norms_square / norms_square


def estimate_projected_operators_sum_of_states(
    states,
    operators,
    n_samples=4032,
    sweep_size=None,
    n_chains=16,
    n_discard_per_chain=5,
    sampling_rule=None,
    include_gram_matrix=True,
    chunk_size=None,
    seed=0,
):
    """
    Estimate the projected operators on a family of variational states using the sum of states estimator.

    Args:
        states (<FullSumState> or <MCState>): The list of states defining the subspace on which to project.
        operators (<LocalOperatorJax>): The operators.
        n_samples (int): The number of samples for the sampler.
        sweep_size (int): The sweep size for the sampler. Defaults to n_sites.
        n_chains (int): The number of chains for the sampler.
        n_discard_per_chain (int): The number of discard per chain for the sampler.
        sampling_rule (MetropolisRule): The rule for the Metropolis sampler. If None and the basis states are MCState, use their rule instead.
        include_gram_matrix (bool): Whether to also return the Gram matrix, which is the projection of the identity.
        chunk_size (int): The size of the chunks of samples to be processed at once.
        seed (int): The seed for the sampler.

    Return:
        <ndarray>: The Rayleigh matrix estimate.
        float: The error on solving the system to get the Rayleigh matrix from the Gram matrix and reduced Hamiltonian.
    """

    m_states = len(states)

    model = states[0].model
    hilbert_space = states[0].hilbert

    if sampling_rule is None:
        if isinstance(states[0], nk.vqs.MCState):
            sampling_rule = states[0].sampler.rule
        else:
            raise ValueError(
                "If the basis states are not MCState, then a sampling rule must be provided."
            )

    if sweep_size is None:
        sweep_size = hilbert_space.size

    sampler = nk.sampler.Metropolis(
        hilbert_space, sampling_rule, n_chains=n_chains, sweep_size=sweep_size
    )

    sum_model = ProbabilitySumState(
        base_network=NothingNet,
        base_arguments=flax.core.freeze({"base_net": model}),
        variable_keys=tuple(states[0].variables.keys()),
        m_states=m_states,
    )

    sum_state = nk.vqs.MCState(sampler, sum_model, seed=jax.random.PRNGKey(seed))
    sum_state.n_samples = n_samples
    sum_state.n_discard_per_chain = n_discard_per_chain

    concatenated_variables = jax.tree.map(
        concatenate_variables, *tuple(state.variables for state in states)
    )
    sum_state.variables = {
        variable_name: {
            "base_arguments_base_net": concatenated_variables[variable_name]
        }
        for variable_name in concatenated_variables
    }

    reduced_observables = estimate_everything(
        sum_state, operators, chunk_size=chunk_size
    )

    return reduced_observables
