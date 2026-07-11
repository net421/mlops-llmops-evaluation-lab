.PHONY: install pipeline test check native-systems shared shared-reproducibility check-shared clean

PYTHON ?= python
SEMANTIC_ROOT ?=
DECISION_TWIN_ROOT ?=
DBT_ROOT ?=
WAREHOUSE_ROOT ?=

install:
	$(PYTHON) -m pip install -e .

pipeline:
	PYTHONPATH=src $(PYTHON) -m mlops_eval.pipeline

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

check: pipeline test

native-systems:
	@test -n "$(SEMANTIC_ROOT)" || (echo "SEMANTIC_ROOT is required" && exit 1)
	@test -n "$(DECISION_TWIN_ROOT)" || (echo "DECISION_TWIN_ROOT is required" && exit 1)
	@test -n "$(DBT_ROOT)" || (echo "DBT_ROOT is required" && exit 1)
	@test -n "$(WAREHOUSE_ROOT)" || (echo "WAREHOUSE_ROOT is required" && exit 1)
	$(MAKE) -C "$(SEMANTIC_ROOT)" verify-live DBT_ROOT="$(DBT_ROOT)" WAREHOUSE_ROOT="$(WAREHOUSE_ROOT)"
	$(MAKE) -C "$(DECISION_TWIN_ROOT)" verify

shared:
	@test -n "$(SEMANTIC_ROOT)" || (echo "SEMANTIC_ROOT is required" && exit 1)
	@test -n "$(DECISION_TWIN_ROOT)" || (echo "DECISION_TWIN_ROOT is required" && exit 1)
	PYTHONPATH=src $(PYTHON) -m mlops_eval.shared_pipeline \
		--semantic-root "$(SEMANTIC_ROOT)" \
		--decision-twin-root "$(DECISION_TWIN_ROOT)" \
		--output-dir artifacts/shared

shared-reproducibility:
	PYTHONPATH=src $(PYTHON) scripts/verify_shared_reproducibility.py \
		--semantic-root "$(SEMANTIC_ROOT)" \
		--decision-twin-root "$(DECISION_TWIN_ROOT)"

check-shared: check native-systems shared shared-reproducibility

clean:
	rm -rf artifacts data/synthetic_customer_risk.csv src/*.egg-info src/mlops_eval/__pycache__ tests/__pycache__
