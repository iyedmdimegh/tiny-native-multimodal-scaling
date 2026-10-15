PYTHON ?= python
TORCH_INDEX = https://download.pytorch.org/whl/cu128

.PHONY: setup data smoke test

setup:
	$(PYTHON) -m pip install -r requirements.txt --extra-index-url $(TORCH_INDEX)
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) scripts/prepare_data.py

smoke:
	$(PYTHON) scripts/run_grid.py --preset smoke

test:
	$(PYTHON) -m pytest -q
