# cosmo_sim_tools

Consolidated toolkit for cosmological simulation analysis, namely Illustris,TNG and BRAHMA.

## Initial layout

- `src/cosmo_sim_tools/illustris/`: for reading Illustrs and TNG data
- `src/cosmo_sim_tools/brahma/`: for reading BRAHMA sims data 
- `src/cosmo_sim_tools/brahma_bh_dynamics/` — Black hole dynamics analysis tools for BRAHMA
- `src/cosmo_sim_tools/arepo_tools/` — analysis wrappers for sims 
- `src/cosmo_sim_tools/extern/` — some helper files
- `src/cosmo_sim_tools/examples/` — runnable scripts
- `notebooks/` — analysis notebooks
- `tests/` — tests

## Installation

### With conda (recommended)

Create the environment from the provided `environment.yml` (run from the repo
root). This installs all dependencies and the package itself in editable mode:

```bash
conda env create -f environment.yml
conda activate cosmo_sim_tools
```

To update the environment after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

### With pip

```bash
pip install -e .
```
