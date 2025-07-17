import os
import sys
import glob

os.environ["NETKET_MPI"] = "0"

import numpy as np
import jax
import flax
import flax.linen as nn

import pickle

import netket as nk
import nqxpack

from tqdm.auto import tqdm

from bridge.models import LinearModel
from bridge._src.bridge_tools import concatenate_variables


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

# Observable sampling parameters
state_n_samples = 2**14 # 2**15
state_n_chains = 2**11 # 2**11
print(f"\nSampling with {state_n_samples} samples and {state_n_chains} chains\n")

# Bridge sampling parameters
n_samples = int(sys.argv[3])
sweep_size = 40
n_chains = 20
n_discard_per_chain = 100

print(f"n_samples={n_samples}")

load_folder_names = [f"h={h_factor}hc/interval_{window_index}" for window_index in windows]
current_folder_name = f"h={h_factor}hc/interval_{windows[-1]}"
bridge_folder_name = current_folder_name + f"/bridge_{n_samples}_{sweep_size}_{n_chains}_{n_discard_per_chain}"
save_filename = bridge_folder_name + f"/observable_values_{state_n_samples}"

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

final_time = (m_states-1) * delta
small_times = np.linspace(0, final_time, m_states)

print(f"Loaded {m_states} states")
print(f"small_times={small_times}")


# Load linear combinations
with open(bridge_folder_name+"/bridge_states", 'rb') as file:
    bridge_states = pickle.load(file)

linear_model = LinearModel(base_network=ReshaperNetwork,
                           base_arguments=flax.core.freeze({'base_net': state_list[0].model}),
                           variable_keys=tuple(state_list[0].variables.keys()),
                           m_states=m_states)


true_sampler = nk.sampler.MetropolisLocal(hilbert_space, n_chains=state_n_chains)

bridge_vstates = []
for bridge_state in bridge_states:
    bridge_vstate = nk.vqs.MCState(true_sampler, linear_model, seed=rng_jax, n_samples=state_n_samples, chunk_size=state_n_chains)
    concatenated_variables = jax.tree.map(concatenate_variables, *tuple(state.variables for state in state_list))
    new_variables = {variable_name: {'base_arguments_base_net': concatenated_variables[variable_name]} for variable_name in concatenated_variables}
    new_variables['params']['coefficients'] = bridge_state
    bridge_vstate.variables = new_variables
    bridge_vstates.append(bridge_vstate)

assert len(bridge_vstates) == len(small_times)

total_x_mean = []
total_x_error = []

energy_mean = []
energy_error = []

with tqdm(total=len(bridge_vstates), file=sys.stdout) as progress_bar:

    for state in bridge_vstates:
        
        res = state.expect(total_x)

        total_x_mean.append(res.mean.item())
        total_x_error.append(res.error_of_mean.item())

        res = state.expect(hamiltonian)

        energy_mean.append(res.mean.item())
        energy_error.append(res.error_of_mean.item())

        progress_bar.update()
        progress_bar.refresh()
        print('\n', end='')



data = (small_times, total_x_mean, total_x_error, energy_mean, energy_error)
data = tuple(np.array(item)[np.newaxis,:] for item in data)
data = np.concatenate(data, axis=0)

np.savetxt(save_filename, data)