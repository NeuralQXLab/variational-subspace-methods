import jax
import jax.numpy as jnp
import flax.linen as nn


class LinearModel(nn.Module):
    """
    A variational state that represents linear combinations of m given states. The states themselves have their parameters fixed.
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
        self.coefficients = self.param(
            "coefficients", nn.initializers.constant(1), (self.m_states,), complex
        )

    def __call__(self, samples):
        batch_shape = samples.shape[:-1]
        repeated_samples = jnp.repeat(
            samples.reshape(-1, 1, samples.shape[-1]), self.m_states, axis=1
        )
        log_amplitudes = self.vmap_network(repeated_samples)
        log_final_amplitudes = jax.scipy.special.logsumexp(
            log_amplitudes, axis=1, b=self.coefficients
        )
        return log_final_amplitudes.reshape(batch_shape)


class NothingNet(nn.Module):
    base_net: nn.Module

    @nn.compact
    def __call__(self, samples):
        return self.base_net(samples)


class ReshaperNetwork(nn.Module):
    base_net: nn.Module

    @nn.compact
    def __call__(self, samples):
        batch_shape = samples.shape[:-1]
        samples = samples.reshape(-1, samples.shape[-1])
        flat_output = self.base_net(samples)
        return flat_output.reshape(batch_shape)
