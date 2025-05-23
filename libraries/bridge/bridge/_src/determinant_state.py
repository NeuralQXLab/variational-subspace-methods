import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import netket as nk

from functools import partial

from netket.sampler import MetropolisRule
from netket.vqs import VariationalState

from bridge._src.bridge_tools import concatenate_variables
from bridge._src.models import NothingNet
from bridge._src.determinant_sampler import DeterminantSampler


class DeterminantModel(nn.Module):
    """The model for a determinant state formed by taking m different variational states over m copies of the base Hilbert space.
    The samples of this state are samples of the extended Hilbert space and have shape (..., n_qubits*m_states).
    """

    base_network: type
    base_arguments: dict
    variable_keys: list
    m_states: int
    n_qubits: int

    def setup(self):
        vmap_network_constructor = nn.vmap(
            self.base_network,
            variable_axes={variable_key: 0 for variable_key in self.variable_keys},
            split_rngs={variable_key: 0 for variable_key in self.variable_keys},
            in_axes=1,
            out_axes=1,
        )
        self.vmap_network = vmap_network_constructor(**self.base_arguments)

    def __call__(self, samples):
        amplitudes_matrix = self.construct_amplitudes_matrix(samples)
        return nk.jax.logdet_cmplx(amplitudes_matrix)

    def call_operator_amplitudes(self, samples, operator):
        """For an operator H, return the log-amplitude of the determinant state where all the basis states have been applied H."""
        operator_amplitudes_matrix = self.construct_operator_amplitudes_matrix(
            samples, operator
        )
        return nk.jax.logdet_cmplx(operator_amplitudes_matrix)

    def construct_amplitudes_matrix(self, samples):
        """
        Construct the m * m matrix whose (i,j) element is $langle s_i | phi_i rangle$.
        These are the actual amplitudes, not the log-amplitudes.
        """
        assert (
            samples.shape[-1] == self.m_states * self.n_qubits
        ), "Samples do not have the correct dimension"
        batch_shape = samples.shape[:-1]
        samples = samples.reshape((-1, 1, self.n_qubits))
        repeated_samples = jnp.repeat(samples, self.m_states, axis=1)
        samples_matrix = jnp.exp(self.vmap_network(repeated_samples))
        return samples_matrix.reshape(*batch_shape, self.m_states, self.m_states)

    def construct_operator_amplitudes_matrix(self, samples, operator):
        """
        Construct the m * m matrix whose (i,j) element is $langle s_i | A | phi_i rangle$ for some operator $A$.
        These are the actual amplitudes, not the log-amplitudes.
        """
        assert (
            samples.shape[-1] == self.m_states * self.n_qubits
        ), "Samples do not have the correct dimension"
        batch_shape = samples.shape[:-1]
        samples = samples.reshape((*batch_shape, self.m_states, self.n_qubits))
        connected_samples, connected_elements = operator.get_conn_padded(samples)
        connected_samples = connected_samples.reshape(-1, 1, self.n_qubits)
        repeated_connected_samples = jnp.repeat(connected_samples, self.m_states, axis=1)
        connected_amplitudes = jnp.exp(self.vmap_network(repeated_connected_samples))
        connected_amplitudes = connected_amplitudes.reshape(*batch_shape, self.m_states, -1, self.m_states)
        return jnp.sum(connected_amplitudes * connected_elements[..., jnp.newaxis], axis=-2)

    def construct_individual_amplitudes(self, samples):
        """Return the amplitudes $langle s_k | phi_k rangle$."""
        assert (
            samples.shape[-1] == self.m_states * self.n_qubits
        ), "Samples do not have the correct dimension"
        batch_shape = samples.shape[:-1]
        samples = samples.reshape((-1, self.m_states, self.n_qubits))
        samples_matrix = jnp.exp(self.vmap_network(samples))
        return samples_matrix.reshape(*batch_shape, self.m_states)

    def construct_amplitude_row(self, samples):
        """Construct the line vector of amplitudes single Hilbert space samples, on which are applied all wavefunctions.
        Meant to be used in the DeterminantSampler.
        """
        assert (
            samples.shape[-1] == self.n_qubits
        ), "Sample does not have the correct dimension"
        batch_shape = samples.shape[:-1]
        samples = samples.reshape(-1, 1, self.n_qubits)
        repeated_sample = jnp.repeat(samples, self.m_states, axis=1)
        amplitudes = jnp.exp(self.vmap_network(repeated_sample))
        return amplitudes.reshape(*batch_shape, self.m_states)


def construct_determinant_state(
    states: list[VariationalState],
    cls: type = None,
    n_samples: int = 4032,
    sweep_size: int | None = None,
    n_chains: int = 16,
    n_discard_per_chain: int = 40,
    sampling_rule: MetropolisRule | None = None,
    chunk_size: int | None = None,
    seed: int = 0,
):
    """
    Construct a determinant state initialized by m states.

    Args:
        states (<FullSumState> or <MCState>): The list of states from which to construct the determinant state.
        cls (type): Either nk.vqs.MCState or nk.vqs.FullSumState. If None, the determinant state is of the type of the first component state.
        n_samples (int): The number of samples for the sampler.
        sweep_size (int): The sweep size for the sampler. Defaults to n_sites * m_states.
        n_chains (int): The number of chains for the sampler.
        n_discard_per_chain (int): The number of discard per chain for the sampler.
        sampling_rule (MetropolisRule): The rule for the Metropolis sampler. If None and the original states are MCState, use their rule instead.
        chunk_size (int): The chunk size for the state.
        seed (int): The seed of the state.

    Return:
        FullSumState or MCState: The determinant state.
    """

    hilbert_space = states[0].hilbert
    n_qubits = hilbert_space.size
    m_states = len(states)

    extended_hilbert_space = nk.hilbert.TensorHilbert(*(hilbert_space,) * m_states)
    model = states[0].model

    determinant_model = DeterminantModel(
        base_network=NothingNet,
        base_arguments=flax.core.freeze({"base_net": model}),
        variable_keys=tuple(states[0].variables.keys()),
        m_states=m_states,
        n_qubits=n_qubits,
    )

    if cls is None:
        cls = type(states[0])
    
    if chunk_size is None:
        chunk_size = states[0].chunk_size

    if issubclass(cls, nk.vqs.FullSumState):

        determinant_state = cls(
            extended_hilbert_space, determinant_model, chunk_size=chunk_size, seed=jax.random.PRNGKey(seed)
        )

    elif issubclass(cls, nk.vqs.MCState):

        if sampling_rule is None:
            if isinstance(states[0], nk.vqs.MCState):
                sampling_rule = states[0].sampler.rule
            else:
                raise ValueError(
                    "If the basis states are not MCState, then a sampling rule must be provided."
                )

        if sweep_size is None:
            sweep_size = hilbert_space.size * m_states

        extended_sampler = DeterminantSampler(
            extended_hilbert_space,
            sampling_rule,
            n_chains=n_chains,
            sweep_size=sweep_size,
        )

        determinant_state = cls(
            extended_sampler,
            determinant_model,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            chunk_size=chunk_size,
            seed=jax.random.PRNGKey(seed),
        )

    else:
        raise TypeError(
            "The class can only be a subclass of either nk.vqs.FullSumState or nk.vqs.MCState."
        )

    concatenated_variables = jax.tree.map(
        concatenate_variables, *tuple(state.variables for state in states)
    )
    determinant_state.variables = {
        variable_name: {
            "base_arguments_base_net": concatenated_variables[variable_name]
        }
        for variable_name in concatenated_variables
    }

    return determinant_state


# Fullsum Rayleigh matrix
def expect_rayleigh_matrix_fullsum(fullsum_det_state, hamiltonian):
    all_states = fullsum_det_state.hilbert.all_states()
    probabilities = fullsum_det_state.probability_distribution()
    return _expect_rayleigh_matrix_fullsum(
        fullsum_det_state.model,
        fullsum_det_state.variables,
        all_states,
        probabilities,
        hamiltonian,
    )


@partial(jax.jit, static_argnames=("model"))
def _expect_rayleigh_matrix_fullsum(
    model, variables, all_states, probabilities, hamiltonian
):
    broadcastable_probabilities = probabilities[:, jnp.newaxis, jnp.newaxis]
    amplitudes_matrix = model.apply(
        variables, all_states, method=model.construct_amplitudes_matrix
    )
    hamiltonian_amplitudes_matrix = model.apply(
        variables,
        all_states,
        hamiltonian,
        method=model.construct_operator_amplitudes_matrix,
    )
    local_energy_matrix = (
        jnp.linalg.inv(amplitudes_matrix) @ hamiltonian_amplitudes_matrix
    )
    return jnp.sum(
        broadcastable_probabilities
        * jnp.where(broadcastable_probabilities == 0, 0, local_energy_matrix),
        axis=0,
    )


def expect_rayleigh_matrix_sampled(mcstate_det_state, hamiltonian, chunk_size=None):
    """Estimate the Rayleigh matrix."""
    samples = mcstate_det_state.samples.reshape((-1, mcstate_det_state.hilbert.size))
    return _expect_rayleigh_matrix_sampled(
        mcstate_det_state.model,
        mcstate_det_state.variables,
        samples,
        hamiltonian,
        chunk_size,
    )


def expect_rayleigh_matrix_sampled_with_samples(
    mcstate_det_state, hamiltonian, samples, chunk_size=None
):
    """Estimate the Rayleigh matrix using the provided samples."""
    samples = samples.reshape((-1, mcstate_det_state.hilbert.size))
    return _expect_rayleigh_matrix_sampled(
        mcstate_det_state.model,
        mcstate_det_state.variables,
        samples,
        hamiltonian,
        chunk_size,
    )


@nk.utils.timing.timed
@partial(
    jax.jit, static_argnames=("model", "chunk_size")
)  # Useless because nk.jax.apply_chunked jits on its own?
def _expect_rayleigh_matrix_sampled(model, variables, samples, hamiltonian, chunk_size):
    """Process samples into an estimation of the Rayleigh matrix."""
    hamiltonian._setup()
    amplitudes_matrix = nk.jax.apply_chunked(
        lambda samples: model.apply(
            variables, samples, method=model.construct_amplitudes_matrix
        ),
        chunk_size=chunk_size,
    )(samples)
    hamiltonian_amplitudes_matrix = nk.jax.apply_chunked(
        lambda samples: model.apply(
            variables,
            samples,
            hamiltonian,
            method=model.construct_operator_amplitudes_matrix,
        ),
        chunk_size=chunk_size,
    )(samples)
    local_energy_matrix = jnp.linalg.solve(
        amplitudes_matrix, hamiltonian_amplitudes_matrix
    )
    return jnp.mean(local_energy_matrix, axis=0)


def construct_component_wavefunctions(variational_state):
    """Construct the component wavefunctions."""
    determinant_model = variational_state.model
    m_states = determinant_model.m_states
    hilbert_space = variational_state.hilbert.subspaces[0]
    all_samples = jnp.concatenate(
        tuple(hilbert_space.all_states() for _ in range(m_states)), axis=1
    )
    return determinant_model.apply(
        variational_state.variables,
        all_samples,
        method=determinant_model.construct_individual_amplitudes,
    )
