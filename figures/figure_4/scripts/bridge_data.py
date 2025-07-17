import os
import sys
import glob

os.environ["NETKET_MPI"] = "0"

import numpy as np
import jax
import flax.linen as nn

import pickle

import netket as nk

import nqxpack

from tqdm.auto import tqdm

import bridge



class ReshaperNetwork(nn.Module):
    base_net: nn.Module
    @nn.compact
    def __call__(self, samples):
        batch_shape = samples.shape[:-1]
        samples = samples.reshape(-1, samples.shape[-1])
        flat_output = self.base_net(samples)
        return flat_output.reshape(batch_shape)


# Window parameters
current_window = int(sys.argv[1])
windows = list(range(1, current_window+1))
print(f"windows={windows}")

# Hamiltonian parameters
J = 1.
hc = 3.044 * J
h_factor = float(sys.argv[2])
h = h_factor * hc
print(f"h_factor={h_factor}")

# Take h_factor and give the associated values
run_param_dict = {0.1: {'delta': 0.05}, 2: {'delta': 0.004106438896}}
delta = run_param_dict[h_factor]['delta']

# Bridge sampling parameters
n_samples = int(sys.argv[3])
sweep_size = 40
n_chains = 20
n_discard_per_chain = 100

print(f"n_samples={n_samples}")

load_folder_names = [f"h={h_factor}hc/interval_{window_index}" for window_index in windows]
current_folder_name = f"h={h_factor}hc/interval_{windows[-1]}"
save_folder_name = current_folder_name + f"/bridge_{n_samples}_{sweep_size}_{n_chains}_{n_discard_per_chain}"
os.mkdir(save_folder_name)

# Graph parameters
grid_width = 8
grid_height = 8
n_qubits = grid_width * grid_height
dims_vec = [n_qubits*[2], n_qubits*[1]]

# Random seed
rng_jax = jax.random.PRNGKey(0)

# Hilbert space
graph = nk.graph.Grid([grid_width, grid_height], pbc=True)
hilbert_space = nk.hilbert.Spin(0.5, n_qubits, inverted_ordering=False)

# Operators
hamiltonian = sum([-J * nk.operator.spin.sigmaz(hilbert_space, i) * nk.operator.spin.sigmaz(hilbert_space, j) for i,j in graph.edges()])
hamiltonian += sum([-h * nk.operator.spin.sigmax(hilbert_space, i) for i in graph.nodes()])
hamiltonian = hamiltonian.to_jax_operator()
total_x = sum([nk.operator.spin.sigmax(hilbert_space, k, dtype=complex).to_pauli_strings().to_jax_operator() for k in range(n_qubits)]) * (1 / n_qubits)


print("Loading states")

# Load states
state_list = []

for load_folder_name in load_folder_names:
    state_file_list = sorted(glob.glob(load_folder_name + "/data/state*"))
    for state_file in tqdm(state_file_list):
        luca_state = nqxpack.load(state_file)
        state = nk.vqs.FullSumState(hilbert_space, ReshaperNetwork(luca_state.model))
        state.variables = {variable_name: {'base_net': luca_state.variables[variable_name]} for variable_name in luca_state.variables}
        state_list.append(state)

m_states = len(state_list)

if m_states == 0:
    raise Exception("Failed to load any state as " + load_folder_name + "/data/state*")

final_time = float(state_file.partition("state_")[2].partition(".nk")[0])
time_steps = round(final_time / delta) + 1
small_times = np.linspace(0, final_time, time_steps)

print(f"Loaded {m_states} states")
print(f"small_times={small_times}")

extended_hilbert_space = nk.hilbert.TensorHilbert(*(hilbert_space,)*m_states)

print("\n----------------------------------------------")
print("| Start bridge for final optimization states |")
print("----------------------------------------------")

sampling_rule = nk.sampler.rules.LocalRule()
bridge_states, rme, info_dict = bridge.bridge(state_list,
                                                hamiltonian,
                                                small_times,
                                                n_samples=n_samples,
                                                sweep_size=sweep_size,
                                                n_chains=n_chains,
                                                n_discard_per_chain=n_discard_per_chain,
                                                sampling_rule=sampling_rule,
                                                chunk_size=1,
                                                )

print("\ninfo_dict:", info_dict)

with open(save_folder_name+"/rayleigh_matrix_estimate", "wb") as file:
    pickle.dump(rme, file)

with open(save_folder_name+"/info_dict", "wb") as file:
    pickle.dump(info_dict, file)

with open(save_folder_name+"/bridge_states", "wb") as file:
    pickle.dump(bridge_states, file)