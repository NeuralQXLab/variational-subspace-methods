import jax
import jax.numpy as jnp
from flax import linen as nn

import netket as nk

from functools import partial


class ProbabilitySumState(nn.Module):
    """
    A variational state that contains parameters of m states such that the probability of a sample is proportional to the sum of the probabilities of the component states.
    """

    base_network: type
    base_arguments: dict
    variable_keys: list
    m_states: int

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
        batch_shape = samples.shape[:-1]
        log_amplitudes = self.construct_log_amplitudes(samples).reshape(
            (-1, self.m_states)
        )
        sum_probabilities = jax.scipy.special.logsumexp(
            2 * jnp.real(log_amplitudes), axis=1
        )
        return sum_probabilities.reshape(batch_shape) / 2

    def construct_log_amplitudes(self, samples):
        batch_shape = samples.shape[:-1]
        samples = samples.reshape((-1, 1, samples.shape[-1]))
        repeated_samples = jnp.repeat(samples, self.m_states, axis=1)
        log_amplitudes = self.vmap_network(repeated_samples)
        return log_amplitudes.reshape(*batch_shape, self.m_states)


@partial(jax.jit, static_argnames=("model", "chunk_size"))
def _estimate_everything_old(model, variables, samples, observables, chunk_size):

    chunked_apply = nk.jax.apply_chunked(
        lambda samples: model.apply(
            variables, samples, method=model.construct_log_amplitudes
        ),
        chunk_size=chunk_size,
    )

    amplitudes = jnp.exp(chunked_apply(samples))
    sum_probabilities = jnp.sum(jnp.abs(amplitudes) ** 2, axis=1)

    # Estimate the Gram matrix
    gram_matrix = jnp.mean(
        jnp.conj(amplitudes)[:, :, jnp.newaxis]
        * amplitudes[:, jnp.newaxis, :]
        / sum_probabilities[:, jnp.newaxis, jnp.newaxis],
        axis=0,
    )
    reduced_observables = [gram_matrix]

    # Estimate each of the observables
    for observable in observables:
        connected_samples, matrix_elements = observable.get_conn_padded(samples)
        connected_amplitudes = jnp.exp(chunked_apply(connected_samples))
        local_estimator_tensor = (
            jnp.conj(amplitudes)[:, jnp.newaxis, :, jnp.newaxis]
            * matrix_elements[:, :, jnp.newaxis, jnp.newaxis]
            * connected_amplitudes[:, :, jnp.newaxis, :]
            / sum_probabilities[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        )
        reduced_observable = jnp.mean(jnp.sum(local_estimator_tensor, axis=1), axis=0)
        reduced_observables.append(reduced_observable)

    return reduced_observables


def estimate_everything(sum_state, observables, chunk_size=None):
    samples = sum_state.samples
    samples = samples.reshape(-1, samples.shape[-1])
    observables = [
        nk.operator.spin.identity(sum_state.hilbert).to_jax_operator()
    ] + observables
    reduced_observables = []
    for observable in observables:
        local_estimator_tensor = _jax_witchcraft(
            sum_state.model, sum_state.variables, samples, observable, chunk_size
        )
        reduced_observables.append(jnp.mean(local_estimator_tensor, axis=0))
    return reduced_observables


@partial(jax.jit, static_argnames=("model", "chunk_size"))
def _jax_witchcraft(model, variables, samples, observable, chunk_size):
    partial_local_estimator_tensor = lambda _samples: _construct_local_estimator_tensor(
        model, variables, _samples, observable
    )
    return nk.jax.apply_chunked(partial_local_estimator_tensor, chunk_size=chunk_size)(
        samples
    )


@partial(jax.jit, static_argnames=("model"))
def _construct_local_estimator_tensor(model, variables, samples, observable):

    amplitudes = jnp.exp(
        model.apply(variables, samples, method=model.construct_log_amplitudes)
    )
    sum_probabilities = jnp.sum(jnp.abs(amplitudes) ** 2, axis=1)

    connected_samples, matrix_elements = observable.get_conn_padded(samples)
    connected_amplitudes = jnp.exp(
        model.apply(variables, connected_samples, method=model.construct_log_amplitudes)
    )
    local_estimator_tensor = (
        jnp.conj(amplitudes)[:, jnp.newaxis, :, jnp.newaxis]
        * matrix_elements[:, :, jnp.newaxis, jnp.newaxis]
        * connected_amplitudes[:, :, jnp.newaxis, :]
        / sum_probabilities[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    )

    return jnp.sum(local_estimator_tensor, axis=1)
