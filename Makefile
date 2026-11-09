.PHONY: install lint format test up down migrate worker eval eval-retrieval eval-offline eval-frameworks

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -v

up:
	docker compose up --build -d

down:
	docker compose down

migrate:
	alembic upgrade head

worker:
	celery -A app.worker.tasks worker --loglevel=INFO --concurrency=2

eval:
	python -m evals

eval-retrieval:
	python -m evals --mode retrieval

eval-offline:
	python -m evals --offline

eval-frameworks:
	python -m evals --frameworks ragas,deepeval --limit 20
