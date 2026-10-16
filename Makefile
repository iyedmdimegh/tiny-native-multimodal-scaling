PYTHON ?= python
TORCH_INDEX = https://download.pytorch.org/whl/cu128

.PHONY: setup data smoke test benchmark lr_sweep grid

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

test:
	$(PYTHON) -m pytest -q
