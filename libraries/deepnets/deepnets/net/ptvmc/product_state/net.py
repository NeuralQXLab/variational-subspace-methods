import jax.numpy as jnp
import flax.linen as nn
import jax

class product_state(nn.Module):
    """
    Network to parametrize a product state.

    Each site j is parametrized by two angles in the Bloch sphere: theta_j and phi_j, such that:
      - For x_j = +1: psi(x_j) = cos(theta_j/2)
      - For x_j = -1: psi(x_j) = sin(theta_j/2) * exp(i * phi_j)

    The overall log wavefunction is then:

        log(psi(x)) = sum_j log( psi(x_j) )

    where psi(x_j) is computed as:

        psi(x_j) = ((1 + x_j)/2) * cos(theta_j/2) +
                   ((1 - x_j)/2) * exp(i * phi_j) * sin(theta_j/2)

    Parameters:
      N: Number of lattice sites (the last dimension of the input).
    """
    N: int  # Number of lattice sites

    def setup(self):
        self.theta = self.param('theta', nn.initializers.normal(), (self.N,))
        self.phi = self.param('phi', nn.initializers.normal(), (self.N,))

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is expected to have shape (..., N) with entries in {-1, +1}.
        # Create parameters for the angles theta and phi (each is a vector of length N).
        theta = self.theta
        phi = self.phi

        # Using broadcasting, theta and phi act only on the last dimension.
        # For each site j:
        #   if x_j = +1, then (1+x_j)/2 = 1 and (1-x_j)/2 = 0, so amplitude = cos(theta_j/2)
        #   if x_j = -1, then (1+x_j)/2 = 0 and (1-x_j)/2 = 1, so amplitude = exp(i*phi_j)*sin(theta_j/2)
        local_amp = (((1 + x) / 2) * jnp.cos(theta / 2) +
                     ((1 - x) / 2) * jnp.exp(1j * phi) * jnp.sin(theta / 2))

        # Compute the logarithm of the local amplitudes elementwise.
        log_local_amp = jnp.log(local_amp + 1e-15)

        # Sum over the last dimension (lattice sites) so that the output shape is the same as the batch dimensions.
        return jnp.sum(log_local_amp, axis=-1)
