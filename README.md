# cosmo_sim_tools

Consolidated toolkit for cosmological simulation analysis.

## Initial layout

- `src/cosmo_sim_tools/illustris/` — canonical illustris reader code (choose one version only)
- `src/cosmo_sim_tools/brahma/` — BRAHMA-specific reader code (from BRAHMA_python)
- `src/cosmo_sim_tools/brahma_bh_dynamics/` — Black hole dynamics analysis tools (from host_galaxies/BH_dynamics_analysis)
- `src/cosmo_sim_tools/arepo_tools/` — analysis wrappers (`arepo_package`, `twoDplot`, `global_props`, `com`, `mdot_to_Lbol`, `cosmology`)
- `src/cosmo_sim_tools/extern/` — small vendored helper utilities
- `src/cosmo_sim_tools/examples/` — runnable scripts
- `notebooks/` — analysis notebooks
- `tests/` — smoke/regression tests

## Migration map (source -> destination)

- `illustris_python_mod/illustris_python_mod/*.py` -> `src/cosmo_sim_tools/illustris/`
- `arepo_package/BRAHMA_python/*.py` -> `src/cosmo_sim_tools/brahma/`
- `host_galaxies/BH_dynamics_analysis/*.py` -> `src/cosmo_sim_tools/brahma_bh_dynamics/`
- `arepo_package/arepo_package.py` -> `src/cosmo_sim_tools/arepo_tools/arepo_package.py`
- `arepo_package/mdot_to_Lbol.py` -> `src/cosmo_sim_tools/arepo_tools/mdot_to_Lbol.py`
- `arepo_package/cosmology.py` -> `src/cosmo_sim_tools/arepo_tools/cosmology.py`
- `packages/arepo_python_tools/twoDplot.py` -> `src/cosmo_sim_tools/arepo_tools/twoDplot.py`
- `packages/arepo_python_tools/global_props.py` -> `src/cosmo_sim_tools/arepo_tools/global_props.py`
- `packages/arepo_python_tools/com.py` -> `src/cosmo_sim_tools/arepo_tools/com.py`
- `packages/ptorrey_packages/utils/calc_hsml.py` -> `src/cosmo_sim_tools/extern/ptorrey_calc_hsml.py`
- `host_galaxies/py_files/run_galaxy2dplot_subhalo.py` -> `src/cosmo_sim_tools/examples/run_galaxy2dplot_subhalo.py`
- `new_brahma_sims/load_brahma_sims.ipynb` -> `notebooks/load_brahma_sims.ipynb`

## Important cleanup during migration

1. Remove all `sys.path.append(...)` and `sys.path.insert(...)` usage.
2. Replace mixed imports (`illustris_python`, `illustris_python_mod`, `BRAHMA_python`) with canonical namespaces:
   - `from cosmo_sim_tools import illustris as il`
   - `from cosmo_sim_tools import brahma`
3. Update local imports to absolute package imports:
   - `from cosmo_sim_tools.arepo_tools.global_props import get_particle_data`
4. Exclude transient files from git:
   - `__pycache__/`, `.ipynb_checkpoints/`, `*.egg-info/`, `slurm_outputs/`

## Editable install

```bash
pip install -e .
```
