.PHONY: install pipeline test check clean

install:
	python -m pip install -e .

pipeline:
	PYTHONPATH=src python -m mlops_eval.pipeline

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

check: pipeline test

clean:
	rm -rf artifacts data/synthetic_customer_risk.csv src/*.egg-info src/mlops_eval/__pycache__ tests/__pycache__

