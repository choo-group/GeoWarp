# GeoWarp
High-performance, differentiable, implicit material point method (MPM) for geomechanics, powered by NVIDIA Warp.

---

## Overview
GeoWarp is an open-source simulation framework tailored for geomechanics, featuring:
- A fully implicit MPM formulation.
- High-performance computing support via [NVIDIA Warp](https://nvidia.github.io/warp/).
- Reverse-mode automatic differentiation (AD).
- Support for both forward simulations and inverse analyses.

## Getting started
### Installation
```bash
pip install -r requirements.txt
```

### How to run
Navigate to a problem folder and execute the corresponding Python script:
```
cd problems/bar_compaction/
python bar_compaction_cpGIMP.py
```

## Code structure
Simulations are organized into four main directories:
- `problems/` contains individual simulation scripts specifying setups.
- `solvers/` includes different MPM solvers for 2D/3D, single- and multi-phase problems.
- `mpm/` includes utility functions for MPM computations.
- `materials/` contains various plastic constitutive models

We also provide a single, standalone file to help new users quickly understand the structure without needing to cross-reference multiple files. Those files are ended with `_everything_in_on_script.py`.

```bash
GeoWarp/
├── problems/                     
│   ├── material_tests/          
│   │   ├── triaxial_DruckerPrager.py
│   │   └── triaxial_NorSand.py
│   ├── bar_compaction/
│   │   └── bar_compaction_cpGIMP.py
│   ├── cantilever_beam/
│   │   └── cantilever_beam_cpGIMP.py
│   ├── consolidation/
│   │   └── terzaghi_cpGIMP.py
│   └── 3D_indentation/
│       └── 3d_indentation_coupled_cpGIMP.py
├── solvers/
│   ├── quasi_static_solver_2d.py               # 2D quasi-static implicit MPM solver 
│   ├── quasi_static_coupled_solver_2d.py       # 2D coupled quasi-static implicit MPM solver
│   ├── quasi_static_coupled_solver_3d.py       # 3D coupled quasi-static implicit MPM solver
│   └── sparse_differentiation.py               # Utility kernels for sparse differentiation
├── mpm/
│   ├── mpm_utils.py                            # Utility kernels for MPM functions
│   └── gimp.py                                 # Utility kernels for generalized interpolation material point shape function
└── materials/
    └── plasticity.py                           # Plastic constitutive models
```

## Citation
If you find GeoWarp useful, please cite us. TODO