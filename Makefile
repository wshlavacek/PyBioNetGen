.PHONY: clean virtualenv test typecheck docker vendor-bng dist dist-upload

clean:
	find . -name '*.py[co]' -delete

virtualenv:
	python -m venv env
	env/bin/python -m pip install -r requirements-dev.txt
	env/bin/python -m pip install -e .
	@echo
	@echo "VirtualENV Setup Complete. Now run: source env/bin/activate"
	@echo

docker: clean
	docker build -t bionetgen:latest .

test:
	python scripts/run_dev_checks.py tests/

typecheck:
	uv run --no-project --with-requirements requirements-dev.txt python -m mypy bionetgen tests

vendor-bng:
	uv run --no-project --with-requirements requirements-dev.txt python scripts/vendor_bionetgen_assets.py

dist: clean vendor-bng
	rm -rf dist build *.egg-info
	uv run --no-project --with-requirements requirements-dev.txt python -m build --sdist --wheel

dist-upload:
	twine upload dist/*
