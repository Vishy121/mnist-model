.PHONY: setup test train clean

setup:
	python -m pip install -e .

test:
	pytest -v tests/

train:
	python train.py

clean:
	rm -rf logs/*
	rm -rf models/*
	rm -rf data/*
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -r {} +

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

all: clean setup test train 