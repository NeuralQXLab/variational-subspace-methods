import jax
import jax.numpy as jnp

import netket as nk

from functools import partial

from netket.sampler.metropolis import _assert_good_sample_shape
from netket.jax.sharding import shard_along_axis

from typing import Any


class DeterminantSamplerState(nk.sampler.MetropolisSamplerState):
    """
    Sampler state for the DeterminantSampler. Contains the current configuration, the RNG state and the (optional) state of the transition rule.
    """

    log_determinant: jax.Array
    amplitude_matrix: jax.Array
    amplitude_matrix_inverse: jax.Array

    def __init__(
        self,
        σ: jnp.ndarray,
        rng: jnp.ndarray,
        rule_state: Any | None,
        log_determinant,
        amplitude_matrix,
        amplitude_matrix_inverse,
    ):

        super().__init__(σ, rng, rule_state)

        self.log_determinant = log_determinant
        self.amplitude_matrix = amplitude_matrix
        self.amplitude_matrix_inverse = amplitude_matrix_inverse


class DeterminantSampler(nk.sampler.MetropolisSampler):
    """Sampler for the determinant state that implements O(m^2) probability update."""

    def _sample_next(sampler, machine, parameters, state):
        """Update the sampler state and the samples for one sweep of the Markov chain."""

        def update_markov_chain(k, state_dict):
            """Perform a steps of the Markov chain."""

            samples = state_dict["samples"]
            n_chains = samples.shape[0]

            # Create the rng for the update
            state_dict["key"], proposition_key, acceptance_key = jax.random.split(
                state_dict["key"], 3
            )
            m_key, rule_key = jax.random.split(proposition_key, 2)

            # Select the component Hilbert space whose configuration will be updated for each chain.
            hilbert_space_indices = jax.random.randint(
                m_key, shape=(n_chains,), minval=0, maxval=machine.m_states
            )

            # Extract the subsample that will be updated for each chain.
            reshaped_samples = samples.reshape(
                n_chains, machine.m_states, machine.n_qubits
            )
            selected_subsamples = reshaped_samples[
                jnp.arange(n_chains), hilbert_space_indices
            ]

            # Update the selected subsample using the transition rule for each chain.
            sub_sampler = sampler.replace(hilbert=sampler.hilbert.subspaces[0])
            updated_selected_subsamples, log_prob_correction = sampler.rule.transition(
                sub_sampler, machine, parameters, state, rule_key, selected_subsamples
            )

            # Construct the updated row of the amplitude matrix.
            updated_selected_rows = machine.apply(
                parameters,
                updated_selected_subsamples,
                method=machine.construct_amplitude_row,
            )

            def update_sample_and_determinant_and_matrices(
                hilbert_space_index,
                sample,
                updated_selected_subsample,
                updated_selected_row,
                previous_amplitude_matrix,
                previous_amplitude_matrix_inverse,
                previous_log_determinant,
            ):
                """Update the sample, determinant, amplitude matrix and inverse amplitude matrix using the matrix determinant lemma and the Sherman-Morrison formula.
                Only works without n_chains axis.
                """

                # Extract the selected row in the previous amplitude matrix.
                previous_selected_row = previous_amplitude_matrix[hilbert_space_index]

                # Construct the correction to the selected row of the amplitude matrix caused by the update.
                correction_row = updated_selected_row - previous_selected_row

                # Construct the updated sample.
                reshaped_sample = sample.reshape(machine.m_states, machine.n_qubits)
                new_sample = (
                    reshaped_sample.at[hilbert_space_index]
                    .set(updated_selected_subsample)
                    .reshape(-1)
                )

                # Update the amplitude matrix.
                updated_amplitude_matrix = previous_amplitude_matrix.at[
                    hilbert_space_index
                ].set(updated_selected_row)

                # Update the inverse amplitude matrix with the Sherman-Morrison formula.
                factor = 1 + jnp.inner(
                    correction_row,
                    previous_amplitude_matrix_inverse[:, hilbert_space_index],
                )
                inverse_numerator = jnp.outer(
                    previous_amplitude_matrix_inverse[:, hilbert_space_index],
                    previous_amplitude_matrix_inverse.T @ correction_row,
                )
                updated_amplitude_matrix_inverse = (
                    previous_amplitude_matrix_inverse - inverse_numerator / factor
                )

                # Update the log determinant with the matrix determinant lemma.
                updated_log_determinant = (
                    jnp.log(factor.astype(complex)) + previous_log_determinant
                )

                return (
                    new_sample,
                    updated_amplitude_matrix,
                    updated_amplitude_matrix_inverse,
                    updated_log_determinant,
                )

            def update_determinant_from_scratch(
                hilbert_space_index,
                sample,
                updated_selected_subsample,
                updated_selected_row,
                previous_amplitude_matrix,
                previous_amplitude_matrix_inverse,
                previous_log_determinant,
            ):
                """Update the sample, determinant, amplitude matrix and inverse amplitude matrix naively.
                Only works without n_chains axis.
                """

                # Extract the selected row in the previous amplitude matrix.
                previous_selected_row = previous_amplitude_matrix[hilbert_space_index]

                # Construct the correction to the selected row of the amplitude matrix caused by the update.
                correction_row = (  # noqa: F841
                    updated_selected_row - previous_selected_row
                )

                # Construct the updated sample.
                reshaped_sample = sample.reshape(machine.m_states, machine.n_qubits)
                new_sample = (
                    reshaped_sample.at[hilbert_space_index]
                    .set(updated_selected_subsample)
                    .reshape(-1)
                )

                # Update the amplitude matrix.
                updated_amplitude_matrix = previous_amplitude_matrix.at[
                    hilbert_space_index
                ].set(updated_selected_row)

                # Update the inverse amplitude matrix.
                updated_amplitude_matrix_inverse = jnp.linalg.inv(
                    updated_amplitude_matrix
                )

                # Update the log determinant.
                updated_log_determinant = nk.jax.logdet_cmplx(updated_amplitude_matrix)

                return (
                    new_sample,
                    updated_amplitude_matrix,
                    updated_amplitude_matrix_inverse,
                    updated_log_determinant,
                )

            # vmapped_update = jax.vmap(update_sample_and_determinant_and_matrices)
            vmapped_update = jax.vmap(update_determinant_from_scratch)
            (
                new_samples,
                new_amplitude_matrix,
                new_amplitude_matrix_inverse,
                new_log_determinant,
            ) = vmapped_update(
                hilbert_space_indices,
                samples,
                updated_selected_subsamples,
                updated_selected_rows,
                state_dict["amplitude_matrix"],
                state_dict["amplitude_matrix_inverse"],
                state_dict["log_determinant"],
            )

            proposal_log_prob = (
                2 * (new_log_determinant - state_dict["log_determinant"]).real
            )

            uniform = jax.random.uniform(acceptance_key, shape=(n_chains,))

            exact_log_amp = machine.apply(parameters, samples[2])  # noqa: F841
            # jax.debug.print("current sample: {}", samples[2])
            # jax.debug.print("current sample tracked log amplitude: {}", state_dict["log_determinant"][2])
            # jax.debug.print("current sample exact log amplitude:   {}", exact_log_amp)
            # jax.debug.print("current sample log amplitude error:   {}", jnp.abs(state_dict["log_determinant"][2].real - exact_log_amp.real))
            # jax.debug.print("is sample invalid: {}", is_invalid_sample(samples[2], machine.m_states))

            if log_prob_correction is not None:
                do_accept = uniform < jnp.exp(proposal_log_prob + log_prob_correction)
            else:
                do_accept = uniform < jnp.exp(proposal_log_prob)

            # jax.debug.print("do_accept={}", do_accept)

            # do_accept must match ndim of proposal and state (which is 2)
            state_dict["samples"] = jnp.where(
                do_accept.reshape(-1, 1), new_samples, samples
            )
            state_dict["amplitude_matrix"] = jnp.where(
                do_accept.reshape(-1, 1, 1),
                new_amplitude_matrix,
                state_dict["amplitude_matrix"],
            )
            state_dict["amplitude_matrix_inverse"] = jnp.where(
                do_accept.reshape(-1, 1, 1),
                new_amplitude_matrix_inverse,
                state_dict["amplitude_matrix_inverse"],
            )
            state_dict["log_determinant"] = jnp.where(
                do_accept, new_log_determinant, state_dict["log_determinant"]
            )
            state_dict["accepted"] += do_accept

            return state_dict

        s = {
            "key": state.rng,
            "samples": state.σ,
            "log_determinant": state.log_determinant,
            "amplitude_matrix": state.amplitude_matrix,
            "amplitude_matrix_inverse": state.amplitude_matrix_inverse,
            "accepted": state.n_accepted_proc,
        }

        # jax.debug.print("\nstep={}", state.n_steps_proc // sampler.n_batches)

        s = jax.lax.fori_loop(0, sampler.sweep_size, update_markov_chain, s)

        new_state = state.replace(
            rng=s["key"],
            σ=s["samples"],
            log_determinant=s["log_determinant"],
            amplitude_matrix=s["amplitude_matrix"],
            amplitude_matrix_inverse=s["amplitude_matrix_inverse"],
            n_accepted_proc=s["accepted"],
            n_steps_proc=state.n_steps_proc + sampler.sweep_size * sampler.n_batches,
        )

        new_state_log_prob = 2 * new_state.log_determinant.real

        return new_state, (new_state.σ, new_state_log_prob)

    @partial(jax.jit, static_argnums=1)
    def _init_state(sampler, machine, parameters, key):
        key_state, key_rule = jax.random.split(key)
        rule_state = sampler.rule.init_state(sampler, machine, parameters, key_rule)
        σ = jnp.zeros((sampler.n_batches, sampler.hilbert.size), dtype=sampler.dtype)
        σ = shard_along_axis(σ, axis=0)

        apply_fun = lambda vars, samples: machine.apply(
            vars, samples, method=machine.construct_amplitudes_matrix
        )
        machine_dtype = jax.eval_shape(apply_fun, parameters, σ).dtype

        log_determinant = jnp.zeros((sampler.n_batches,), dtype=complex)
        log_determinant = shard_along_axis(log_determinant, axis=0)

        amplitude_matrix = jnp.zeros(
            (sampler.n_batches, machine.m_states, machine.m_states), dtype=machine_dtype
        )
        amplitude_matrix = shard_along_axis(amplitude_matrix, axis=0)

        amplitude_matrix_inverse = jnp.zeros(
            (sampler.n_batches, machine.m_states, machine.m_states), dtype=machine_dtype
        )
        amplitude_matrix_inverse = shard_along_axis(amplitude_matrix_inverse, axis=0)

        state = DeterminantSamplerState(
            σ=σ,
            rng=key_state,
            rule_state=rule_state,
            log_determinant=log_determinant,
            amplitude_matrix=amplitude_matrix,
            amplitude_matrix_inverse=amplitude_matrix_inverse,
        )

        # If we don't reset the chain at every sampling iteration, then reset it
        # now.
        if not sampler.reset_chains:
            key_state, rng = jax.jit(jax.random.split)(key_state)
            # σ = sampler.rule.random_state(sampler, machine, parameters, state, rng)
            σ = random_state_not_prob_0(
                sampler.hilbert,
                rng,
                sampler.n_batches,
                dtype=sampler.dtype,
                m_states=machine.m_states,
            )
            _assert_good_sample_shape(
                σ,
                (sampler.n_batches, sampler.hilbert.size),
                sampler.dtype,
                f"{sampler.rule}.random_state",
            )
            σ = shard_along_axis(σ, axis=0)
            state = state.replace(σ=σ, rng=key_state)
        return state

    @partial(jax.jit, static_argnums=1)
    def _reset(self, machine, parameters, state):

        state = super()._reset(machine, parameters, state)

        state = state.replace(
            log_determinant=machine.apply(parameters, state.σ),
            amplitude_matrix=machine.apply(
                parameters, state.σ, method=machine.construct_amplitudes_matrix
            ),
            amplitude_matrix_inverse=jnp.linalg.inv(
                machine.apply(
                    parameters, state.σ, method=machine.construct_amplitudes_matrix
                )
            ),
        )

        # jax.debug.print("{}", state.log_determinant[2])
        # jax.debug.print("initial sample: {}", state.σ[2])

        return state


def random_state_not_prob_0(hilb, key, batches: int, *, dtype=None, m_states):

    keys = jax.random.split(key, batches + 1)
    states = hilb.random_state(keys[0], batches, dtype=dtype)

    def _loop_until_ok(state, key):
        def __body(args):
            state, _key = args
            _key, subkey = jax.random.split(_key)
            new_state = hilb.random_state(subkey, dtype=dtype)
            return (new_state, _key)

        def __cond(args):
            state, _ = args
            return is_invalid_sample(state, m_states)

        return jax.lax.while_loop(__cond, __body, (state, key))[0]

    return jax.vmap(_loop_until_ok, in_axes=(0, 0))(states, keys[1:])


def is_invalid_sample(sample, m_states):
    """Check whether a sample has duplicate subconfiguration and therefore has probability 0."""
    assert len(sample.shape) == 1
    sample = sample.reshape(m_states, -1)
    comparison_array = (
        sample[:, jnp.newaxis, :]
        - sample[jnp.newaxis, :, :]
        + jnp.eye(m_states, m_states)[:, :, jnp.newaxis]
    )
    return jnp.any(jnp.all(comparison_array == 0, axis=2))
