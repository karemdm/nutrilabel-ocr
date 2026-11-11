.PHONY: install install-all lint format test cov benchmark report demo ui docker clean

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all,dev]"

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

cov:
	pytest --cov=nutrilabel --cov-report=term-missing

benchmark:
	python -m benchmark.run

report:
	python -m benchmark.report --gallery 6

demo:
	python scripts/make_demo.py

ui:
	nutrilabel ui

docker:
	docker build -t nutrilabel .

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
