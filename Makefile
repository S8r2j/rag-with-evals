.PHONY: install lint format test up down migrate

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
