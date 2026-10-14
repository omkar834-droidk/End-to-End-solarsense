.PHONY: install train test lint run up
install:
	pip install -r requirements-dev.txt
train:
	python scripts/train.py
test:
	pytest -q
lint:
	ruff check src tests scripts
run:
	PYTHONPATH=src uvicorn solarsense.api.main:app --reload
up:
	docker compose up --build
