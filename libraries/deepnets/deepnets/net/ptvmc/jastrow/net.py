import jax.numpy as jnp

from netket.utils.types import Any, NNInitFunc

import jax
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
from jax.nn.initializers import normal


def vec_to_tril(kernel, shape, indices):
    r"""
    Maps a kernel vector to a lower-triangular matrix of indicated shape.
    This function basically does jnp.zeros(shape).at[indices].set(kernel) in an optimized way
    Args:
        kernel: vector with given parameters
        shape: resulting shape of the matrix
        indices: mapping between kernel and final matrix
    Returns:
        parameter matrix with `kernel` at `indices` and zeros otherwise.
    """

    if jnp.issubdtype(kernel.dtype, jnp.complex128):
        Wr = (
            jnp.zeros(shape, dtype=jnp.float64)
            .at[indices]
            .set(kernel.real, unique_indices=True, indices_are_sorted=True)
        )
        Wi = (
            jnp.zeros(shape, dtype=jnp.float64)
            .at[indices]
            .set(kernel.imag, unique_indices=True, indices_are_sorted=True)
        )
        W = Wr + 1j * Wi

    else:
        W = (
            jnp.zeros(shape, dtype=kernel.dtype)
            .at[indices]
            .set(kernel, unique_indices=True, indices_are_sorted=True)
        )
    return W


class JasMultipleBody(nn.Module):
    r"""
    n-body Jastrow implementation, as the n-body matrix is represented as a product of two bodies:
    ..math::
        W_ijkl... = W_ij W_jk W_kl ...

    where each :math:`W` is a 2-body lower triangular matrix. The log_value is then obtained as
    ..math::
        logpsi(z) = \sum_i zi \sum_j W_ij z_j \sum_k W_jk z_k \sum_l W_kl z_l ...

    Warning: do not initialize at 0, otherwise the gradient are also 0 and you get stuck
    """

    n: int = 2
    """The number of interactions."""

    param_dtype: Any = jnp.complex128
    """The dtype of the weights."""

    kernel_init: NNInitFunc = normal(1e-2)
    """Initializer for the jastrow parameters."""

    @nn.compact
    def __call__(self, x_in):
        """
        x : (Ns,N)
        """
        N = x_in.shape[-1]

        # 2 bodies jastrow
        kernel = self.param(
            "kernel", self.kernel_init, (N * (N - 1) // 2,), self.param_dtype
        )
        il = jnp.tril_indices(N, k=-1)
        W = vec_to_tril(kernel, (N, N), il)

        W, x_in = promote_dtype(W, x_in, dtype=None)

        # initialize
        z = jnp.einsum("ni,ji->nj", x_in, W)

        z = jax.lax.fori_loop(
            0, self.n - 2, lambda i, z: jnp.einsum("nj,kj,nj->nk", z, W, x_in), z
        )

        return jnp.einsum("nl,nl->n", z, x_in)


class Jastrow(nn.Module):
    """
    A network that returns the sum of multiple n-body Jastrow terms.

    Attributes:
      orders: A tuple of integers indicating which orders (2-body, 3-body, etc.)
              to include. For example, (2, 3, 4) will include 2-body, 3-body, and 4-body terms.
      param_dtype: The data type of the parameters (default: jnp.complex128).
      kernel_init: The initializer function for the Jastrow parameters.
    """

    orders: tuple
    param_dtype: Any = jnp.complex128
    kernel_init: NNInitFunc = normal(1e-2)

    def setup(self):
        orders = tuple(sorted(set(self.orders)))

        self.jas_modules = [
            JasMultipleBody(
                n=order,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                name=f"jas_{order}",  # explicit naming to avoid parameter collisions
            )
            for order in orders
        ]

    def __call__(self, x_in):
        output = 0.0
        for jas in self.jas_modules:
            output += jas(x_in)
        return output
