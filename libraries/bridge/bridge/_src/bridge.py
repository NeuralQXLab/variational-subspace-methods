import numpy as np
import jax
import flax

import mpmath as mp

import netket as nk

from time import time

from netket.operator import DiscreteJaxOperator
from netket.sampler import MetropolisRule
from netket.vqs import VariationalState, MCState

from bridge._src.bridge_tools import concatenate_variables
from bridge._src.models import NothingNet
from bridge._src.determinant_state import (
    construct_determinant_state,
    expect_rayleigh_matrix_sampled,
)
from bridge._src.sum_of_states import ProbabilitySumState, estimate_everything


@nk.utils.timing.timed
def estimate_rayleigh_matrix_determinant_state(
    states: list[VariationalState],
    hamiltonian: DiscreteJaxOperator,
    cls: type = MCState,
    n_samples: int = 4032,
    sweep_size: int | None = None,
    n_chains: int = 16,
    n_discard_per_chain: int = 40,
    sampling_rule: MetropolisRule | None = None,
    chunk_size: int | None = None,
    seed: int = 0,
):
    """
    Estimate the Rayleigh matrix of a family of variational states using the determinant estimator.

    Args:
        states (<FullSumState> or <MCState>): The list of states from which to estimate the Rayleigh matrix.
        hamiltonian (LocalOperatorJax): The Hamiltonian.
        cls (type): Either :class:`nk.vqs.MCState` or :class:`nk.vqs.FullSumState`. If None, the determinant state is of the type of the first component state. We emphasize that if the exact Rayleigh matrix is desired, this is an extremely inefficient way to obtain it.
        n_samples (int): The number of samples for the sampler.
        sweep_size (int): The sweep size for the sampler. Defaults to n_sites * m_states.
        n_chains (int): The number of chains for the sampler.
        n_discard_per_chain (int): The number of discard per chain for the sampler.
        sampling_rule (MetropolisRule): The rule for the Metropolis sampler. If None and the basis states are MCState, use their rule instead.
        chunk_size (int): The size of the chunks of samples to be processed at once.
        seed (int): The seed for the sampler.
        sampling_rule (?): The sampling rule for the determinant sampler.

    Return:
        ndarray: The Rayleigh matrix estimate.
    """

    determinant_state = construct_determinant_state(
        states,
        cls=cls,
        n_samples=n_samples,
        sweep_size=sweep_size,
        n_chains=n_chains,
        n_discard_per_chain=n_discard_per_chain,
        sampling_rule=sampling_rule,
        seed=seed,
    )

    rayleigh_matrix_estimate = expect_rayleigh_matrix_sampled(
        determinant_state, hamiltonian, chunk_size=chunk_size
    )

    return np.array(rayleigh_matrix_estimate)


@nk.utils.timing.timed
def estimate_rayleigh_matrix_sum_of_states(
    states,
    hamiltonian,
    n_samples=4032,
    sweep_size=None,
    n_chains=16,
    n_discard_per_chain=5,
    sampling_rule=None,
    gram_rcond=None,
    gram_decimal_precision=None,
    chunk_size=None,
    seed=0,
):
    """
    Estimate the Rayleigh matrix of a family of variational states using the sum of states estimator.

    Args:
        states (<FullSumState> or <MCState>): The list of states from which to estimate the Rayleigh matrix.
        hamiltonian (LocalOperatorJax): The Hamiltonian.
        n_samples (int): The number of samples for the sampler.
        sweep_size (int): The sweep size for the sampler. Defaults to n_sites.
        n_chains (int): The number of chains for the sampler.
        n_discard_per_chain (int): The number of discard per chain for the sampler.
        sampling_rule (MetropolisRule): The rule for the Metropolis sampler. If None and the basis states are MCState, use their rule instead.
        gram_rcond (int): If estimation_method is sum_of_state, specify the rcond for taking the pseudo-inverse of the Gram matrix. If None, pseudo-inverse is not used.
        gram_decimal_precision (int): If estimation_method is sum_of_state and gram_rcond is None, specify the precision to invert the Gram matrix.
        chunk_size (int): The size of the chunks of samples to be processed at once.
        seed (int): The seed for the sampler.

    Return:
        ndarray: The Rayleigh matrix estimate.
        float: The error on solving the system to get the Rayleigh matrix from the Gram matrix and reduced Hamiltonian.
    """

    model = states[0].model
    hilbert_space = states[0].hilbert

    m_states = len(states)

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
        sum_state, [hamiltonian], chunk_size=chunk_size
    )

    if gram_rcond is not None:
        print("Using pinv")
        rayleigh_matrix_estimate = (
            np.linalg.pinv(reduced_observables[0], rcond=gram_rcond)
            @ reduced_observables[1]
        )
        gram_inverse_error = (
            np.linalg.norm(
                reduced_observables[0] @ rayleigh_matrix_estimate
                - reduced_observables[1]
            )
            / rayleigh_matrix_estimate.size
        )
    elif gram_decimal_precision is not None:
        print("Using mpmath inverse")
        with mp.workdps(gram_decimal_precision):
            gram_matrix_mp = mp.matrix(reduced_observables[0])
            reduced_hamiltonian_mp = mp.matrix(reduced_observables[1])
            rayleigh_matrix_estimate_mp = (
                gram_matrix_mp**-1 @ reduced_hamiltonian_mp
            )  # Should use qr_solve but it is only designed for when b is a vector
            gram_inverse_error = float(
                mp.norm(
                    gram_matrix_mp @ rayleigh_matrix_estimate_mp
                    - reduced_hamiltonian_mp
                )
                / m_states**2
            )
            rayleigh_matrix_estimate = np.array(
                rayleigh_matrix_estimate_mp, dtype=complex
            ).reshape(m_states, m_states)
    else:
        print("Using numpy solve")
        rayleigh_matrix_estimate = np.linalg.solve(
            reduced_observables[0], reduced_observables[1]
        )
        gram_inverse_error = (
            np.linalg.norm(
                reduced_observables[0] @ rayleigh_matrix_estimate
                - reduced_observables[1]
            )
            / rayleigh_matrix_estimate.size
        )

    return np.array(rayleigh_matrix_estimate), gram_inverse_error


@nk.utils.timing.timed
def solve_reduced_equation_np(
    effective_hamiltonian, times, discard_imaginary, initial_state_index=0
):
    """
    Solve the Schrodinger equation for the given effective Hamiltonian with the initial state being the first state of the basis. The equation is solved by diagonalizing the Hamiltonian
    and taking its exponential at different times using the eigendecomposition. The imaginary part of the eigenvalues of the effective Hamiltonian is removed by default.

    Args:
        effective_hamiltonian (ndarray): The effective Hamiltonian for the Schrodinger equation.
        times (ndarray): The list of times at which to evaluate the solution.
        discard_imaginary (bool): Whether to discard the imaginary part of the eigenvalues of the effective Hamiltonian.

    Return:
        ndarray: The solution of the equation at all times with shape (n_times, n_dims).
        dict: The errors of the solver with the following fields:
            eigendecomposition_error: The error on the eigendecomposition computed as M @ U - U @ Λ.
            solve_error: The error on the solution of the system to obtain the initial state in the eigenbasis.
            eigvals_imag: The norm of the imaginary part of the eigenvalues of the effective Hamiltonian.
    """

    times -= times[0]

    dynamics_error_dict = {}

    initial_state = np.zeros(effective_hamiltonian.shape[0], dtype=complex)
    initial_state[initial_state_index] = 1

    eigvals, eigvecs = np.linalg.eig(effective_hamiltonian)

    dynamics_error_dict["eigendecomposition_error"] = (
        np.linalg.norm(effective_hamiltonian @ eigvecs - eigvecs @ np.diag(eigvals))
        / effective_hamiltonian.size
    )
    dynamics_error_dict["eigvals_imag"] = (
        np.linalg.norm(np.imag(eigvals)) / eigvals.size
    )

    if discard_imaginary:
        eigvals = np.real(eigvals)

    rotated_initial_state = np.linalg.solve(eigvecs, initial_state)
    dynamics_error_dict["solve_error"] = (
        np.linalg.norm(eigvecs @ rotated_initial_state - initial_state)
        / initial_state.size
    )

    diagonal = np.exp(-1.0j * times[:, np.newaxis] * eigvals[np.newaxis, :])
    bridge_states = np.einsum("ij,tj,j->ti", eigvecs, diagonal, rotated_initial_state)

    return bridge_states, dynamics_error_dict


@nk.utils.timing.timed
def solve_reduced_equation_mp(
    effective_hamiltonian,
    times,
    decimal_precision,
    discard_imaginary,
    initial_state_index=0,
):
    """
    Solve the Schrodinger equation for the given effective Hamiltonian with the initial state being the first state of the basis. The equation is solved by diagonalizing the Hamiltonian
    and taking its exponential at different times using the eigendecomposition. The imaginary part of the eigenvalues of the effective Hamiltonian is removed by default.
    This version solves the equation with arbitrary precision using mpmath.

    Args:
        effective_hamiltonian (ndarray): The effective Hamiltonian for the Schrodinger equation.
        times (ndarray): The list of times at which to evaluate the solution.
        decimal_precision (int): The decimal precision for solving the equation.
        discard_imaginary (bool): Whether to discard the imaginary part of the eigenvalues of the effective Hamiltonian.
        initial_state_index (int): The index of the initial state in the list of state used to construct the effective Hamiltonian.

    Return:
        ndarray: The solution of the equation at all times with shape (n_times, n_dims).
        dict: The errors of the solver with the following fields:
            eigendecomposition_error: The error on the eigendecomposition computed as M @ U - U @ Λ.
            solve_error: The error on the solution of the system to obtain the initial state in the eigenbasis.
            eigvals_imag: The norm of the imaginary part of the eigenvalues of the effective Hamiltonian.
    """

    with mp.workdps(decimal_precision):

        dynamics_error_dict = {}

        n_steps = len(times)
        dim = np.shape(effective_hamiltonian)[0]

        initial_state = mp.zeros(dim, 1)
        initial_state[initial_state_index, 0] = 1

        times_mp = mp.matrix(times - times[0])
        effective_hamiltonian_mp = mp.matrix(effective_hamiltonian)

        eigvals, eigvecs = mp.eig(effective_hamiltonian_mp)
        eigvals = mp.matrix(eigvals)

        dynamics_error_dict["eigendecomposition_error"] = float(
            mp.norm(effective_hamiltonian_mp @ eigvecs - eigvecs @ mp.diag(eigvals))
            / dim**2
        )
        dynamics_error_dict["eigvals_imag"] = float(mp.norm(eigvals.apply(mp.im)) / dim)

        if discard_imaginary:
            eigvals = eigvals.apply(mp.re)

        rotated_initial_state = mp.qr_solve(eigvecs, initial_state)[0]
        dynamics_error_dict["solve_error"] = float(
            mp.norm(eigvecs @ rotated_initial_state - initial_state) / dim
        )

        time_diagonals = (-1.0j * times_mp * eigvals.T).apply(
            mp.exp
        )  # shape (n_steps, dim)
        for k in range(dim):
            time_diagonals[:, k] *= rotated_initial_state[k]
        solutions = time_diagonals * eigvecs.T
        bridge_states = np.array(solutions, dtype=complex).reshape(n_steps, dim)

        return bridge_states, dynamics_error_dict


def bridge(
    states: list[VariationalState],
    hamiltonian: DiscreteJaxOperator,
    times: np.ndarray,
    n_samples: int = 4032,
    sweep_size: int | None = None,
    n_chains: int = 16,
    n_discard_per_chain: int = 40,
    sampling_rule: MetropolisRule = None,
    chunk_size: int | None = None,
    estimation_method: str = "determinant_state",
    decimal_precision_solver: int = 100,
    discard_imaginary: bool = False,
    gram_rcond: float | None = None,
    gram_decimal_precision: float | None = None,
    seed: int = 0,
    timeit: bool = True,
    initial_state_index: int = 0,
):
    """
    Perform bridge.

    Args:
        states (<FullSumState> or <MCState>): The list of states to use as a basis for bridge.
        hamiltonian (LocalOperatorJax): The Hamiltonian.
        times (ndarray): The list of times at which to get the bridge states.
        n_samples (int): The number of samples for the sampler.
        sweep_size (int): The sweep size for the sampler.
        n_chains (int): The number of chains for the sampler.
        n_discard_per_chain (int): The number of discard per chain for the sampler.
        sampling_rule (MetropolisRule): The rule for the Metropolis sampler. If None and the basis states are MCState, use their rule instead.
        chunk_size (int): The size of the chunks of samples to be processed at once.
        estimation_method (str): Specify which method to use for the estimation of the Rayleigh matrix. Must be either 'determinant_state' or 'sum_of_states'.
        decimal_precision_solver (int): The decimal precision for solving the effective Schrödinger equation with mpmath. If None, solve using double precision in NumPy.
        discard_imaginary (bool): Whether to discard the imaginary part of the eigenvalues of the estimated Rayleigh matrix.
        gram_rcond (int): If estimation_method is sum_of_state, specify the rcond for taking the pseudo-inverse of the Gram matrix. If None, pseudo-inverse is not used.
        gram_decimal_precision (int): If estimation_method is sum_of_state and gram_rcond is None, specify the precision to invert the Gram matrix.
        seed (int): The seed for the sampler of the Rayleigh matrix.
        timeit (bool): Whether to display the timing information.
        initial_state_index (int): The index of the state in the states list that should be taken as the initial state of the evolution.

    Return:
        ndarray: The result of bridge at all times with shape (n_times, n_states) in the basis of the given states.
        ndarray: The Rayleigh matrix estimate.
        dict: A dictionary containing information on the run.
    """

    with nk.utils.timing.timed_scope(force=timeit) as timer:

        info_dict = {}

        rme_initial_time = time()
        if estimation_method == "determinant_state":
            rayleigh_matrix_estimate = estimate_rayleigh_matrix_determinant_state(
                states,
                hamiltonian,
                n_samples=n_samples,
                sweep_size=sweep_size,
                n_chains=n_chains,
                n_discard_per_chain=n_discard_per_chain,
                sampling_rule=sampling_rule,
                chunk_size=chunk_size,
                seed=seed,
            )
        elif estimation_method == "sum_of_states":
            rayleigh_matrix_estimate, gram_inverse_error = (
                estimate_rayleigh_matrix_sum_of_states(
                    states,
                    hamiltonian,
                    n_samples,
                    sweep_size,
                    n_chains,
                    n_discard_per_chain,
                    sampling_rule,
                    gram_rcond,
                    gram_decimal_precision,
                    chunk_size,
                    seed,
                )
            )
            info_dict["gram_inverse_error"] = gram_inverse_error
        else:
            raise ValueError(
                f"Estimation method should be either 'determinant_state' or 'sum_of_states'. Received '{estimation_method}'."
            )
        info_dict["rme_time"] = time() - rme_initial_time

        try:
            if decimal_precision_solver is None:
                bridge_states, dynamics_error_dict = solve_reduced_equation_np(
                    rayleigh_matrix_estimate,
                    np.array(times),
                    discard_imaginary=discard_imaginary,
                    initial_state_index=initial_state_index,
                )
            else:
                bridge_states, dynamics_error_dict = solve_reduced_equation_mp(
                    rayleigh_matrix_estimate,
                    np.array(times),
                    decimal_precision=decimal_precision_solver,
                    discard_imaginary=discard_imaginary,
                    initial_state_index=initial_state_index,
                )
        except BaseException as error:
            print("Error raised during equation solving:", error)
            print("Returning Rayleigh matrix estimate.")
            return None, rayleigh_matrix_estimate, info_dict
        info_dict.update(dynamics_error_dict)

    if timeit:
        print(timer)

    return bridge_states, rayleigh_matrix_estimate, info_dict
