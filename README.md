# Variational subspace methods
[![arXiv](https://img.shields.io/badge/arXiv-2507.05352-b31b1b.svg)](https://arxiv.org/abs/2507.08930)

This repository holds the accompanying code for the paper '[Variational subspace methods and application to improving
variational Monte Carlo dynamics](https://arxiv.org/abs/2507.08930)', organised into the `bridge` library implementing the variational dynamics post-processing method explored in the manuscript, as well as examples on how to use it, and the data and scripts corresponding to the figures of the paper.

## Structure

- `examples`: Examples of how to use bridge.
    - `bridge_example`: A simple example using first order discretized time evolution states as a basis.
    - `bridge_4x4`: An example of using bridge on a $4 \times 4$ lattice using p-tVMC states as a basis.
- `figures`: The data and scripts required to reproduce the figures of the paper. For each figure, it contains the basis states that were used for bridge, the script to run bridge, the data of the simulation, and the script to plot this data.
- `libraries`:
    - `bridge`: The library that implements bridge. It also contains tools to create and manipulate the determinant state described in section 2 of the paper.
    - The other libraries serve to define the networks used in the paper and ensure that the basis states can be loaded.

## Installation

This package is not registered on PyPi, so you must install it directly from GitHub. To do so, you can run the following command:
```bash
pip install git+https://github.com/NeuralQXLab/variational-subspace-methods
```

You can also clone the repository and install the package locally by running
```bash
git clone https://github.com/NeuralQXLab/variational-subspace-methods
cd variational-subspace-methods
pip install -e .
```

In order to run with GPU, you should add `"jax[cuda]"` to your environment.

## Cite

If you use this code in your work, please cite the associated paper:
```
@article{Kahn2025Subspaces,
      title={Variational subspace methods and application to improving variational Monte Carlo dynamics}, 
      author={Adrien Kahn and Luca Gravina and Filippo Vicentini},
      year={2025},
      eprint={2507.08930},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2507.08930}, 
}
```

## Further information
For further questions, you can contact adrien.kahn.x19@polytechnique.edu.
