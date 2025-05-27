import jax
import jax.numpy as jnp
import numpy as np

import flax
import flax.linen as nn

from netket.utils.types import DType


def _todense(x):
    if hasattr(x, "todense"):
        return x.todense()
    else:
        return x


class Jastrow_zz_frozen(nn.Module):
    network: nn.Module
    """The network containig the correlated state."""

    kernel_zz_init: jnp.array = jnp.zeros
    """The zz Jastrow network."""

    param_dtype: DType = jnp.complex128
    """dtype for parameters."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        N = x.shape[-1]

        kernel_zz = self.variable(
            "modifiers",
            "kernel_zz",
            self.kernel_zz_init,
            (N * (N - 1) // 2,),
            self.param_dtype,
        )

        il = jnp.tril_indices(N, k=-1)
        if jnp.issubdtype(self.param_dtype, jnp.complex128):
            W_zz_r = (
                jnp.zeros((N, N), dtype=self.param_dtype)
                .at[il]
                .set(kernel_zz.value.real, unique_indices=True, indices_are_sorted=True)
            )
            W_zz_i = (
                jnp.zeros((N, N), dtype=self.param_dtype)
                .at[il]
                .set(kernel_zz.value.imag, unique_indices=True, indices_are_sorted=True)
            )
            W_zz = W_zz_r + 1j * W_zz_i

        else:
            W_zz = (
                jnp.zeros((N, N), dtype=self.param_dtype)
                .at[il]
                .set(kernel_zz.value, unique_indices=True, indices_are_sorted=True)
            )

        output_zz = -1j * jnp.einsum("...i,ij,...j", x, W_zz, x)

        output_nn = self.network(x)

        return output_nn + output_zz

    @nn.nowrap
    def apply_zz(self, variables, H_diagonal, scale: float = 1.0):
        """
        Applies zz to the parameters of state_modyfiers.

        Args:
            variables: the dictionary of variables.
            H_diagonal: the diagonal part of the Hamiltonian (a LocalOperator).
            scale: the scaling factor for the zz kernel.
        """

        indices = H_diagonal.acting_on
        operators = H_diagonal.operators
        op_coefficients = np.array(
            [_todense(o)[0, 0] for o, t in zip(operators, indices) if len(t) > 1]
        )

        i, j = np.array([t for t in indices if len(t) == 2]).T
        lin_indices = lin_to_tril_index(i, j)

        assert op_coefficients.shape == lin_indices.shape

        kernel = variables["modifiers"]["kernel_zz"]
        if not isinstance(kernel, jax.Array):
            kernel = jnp.asarray(kernel)

        kernel = kernel.at[lin_indices].add(scale * op_coefficients)
        new_variables = flax.core.copy(variables, {"modifiers": {"kernel_zz": kernel}})

        return new_variables


def lin_to_tril_index(i_array, j_array):
    swapped = i_array < j_array
    i_array_swapped = np.where(swapped, j_array, i_array)
    j_array_swapped = np.where(swapped, i_array, j_array)

    return i_array_swapped * (i_array_swapped - 1) // 2 + j_array_swapped
