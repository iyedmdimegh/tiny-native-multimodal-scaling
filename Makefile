PYTHON ?= python
TORCH_INDEX = https://download.pytorch.org/whl/cu128

.PHONY: setup data smoke test benchmark lr_sweep grid fit controls mixture fusion compute figures

setup:
	$(PYTHON) -m pip install -r requirements.txt --extra-index-url $(TORCH_INDEX)
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) scripts/prepare_data.py

smoke:
	$(PYTHON) scripts/run_grid.py --preset smoke

benchmark:
	$(PYTHON) scripts/benchmark.py

lr_sweep:
	$(PYTHON) scripts/run_lr_sweep.py --preset full

grid:
	$(PYTHON) scripts/run_grid.py --preset full

fit:
	$(PYTHON) scripts/fit_scaling.py

RUNS ?= $(or $(TNMS_RUNS_DIR),runs)

controls:
	$(PYTHON) scripts/run_controls.py overfit
	$(PYTHON) scripts/run_controls.py shuffled
	$(PYTHON) scripts/run_controls.py ablation $(RUNS)/full/S4_D80000000_seed0
	$(PYTHON) scripts/run_controls.py gap
	$(PYTHON) scripts/run_controls.py samples $(RUNS)/full/S5_D80000000_seed0

mixture:
	$(PYTHON) scripts/run_mixture.py sweep
	$(PYTHON) scripts/fit_mixture.py
	$(PYTHON) scripts/run_mixture.py validate

fusion:
	$(PYTHON) scripts/run_fusion.py

compute:
	$(PYTHON) scripts/compute_used.py

figures:
	$(PYTHON) scripts/make_figures.py

test:
	$(PYTHON) -m pytest -q
