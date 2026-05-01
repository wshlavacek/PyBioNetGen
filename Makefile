.PHONY: clean virtualenv test typecheck docker dist dist-upload

clean:
	find . -name '*.py[co]' -delete

virtualenv:
	virtualenv --prompt '|> bionetgen <| ' env
	env/bin/pip install -r requirements-dev.txt
	env/bin/python setup.py develop
	@echo
	@echo "VirtualENV Setup Complete. Now run: source env/bin/activate"
	@echo

docker: clean
	docker build -t bionetgen:latest .

test:
	uv run --no-project --with-requirements requirements-dev.txt python scripts/run_dev_checks.py tests/

typecheck:
	uv run --no-project --with-requirements requirements-dev.txt python -m mypy bionetgen tests

dist: clean
	rm -rf dist/*
	python setup.py sdist
	python setup.py bdist_wheel

dist-upload:
	twine upload dist/*
