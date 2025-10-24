.PHONY: help install install-dev clean test lint format run-streamlit run-api docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make test           - Run tests with coverage"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with black and isort"
	@echo "  make run-streamlit  - Run Streamlit application"
	@echo "  make run-api        - Run FastAPI application"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Cleaning
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/

# Testing
test:
	pytest tests/ -v --cov=src/sentiment_analysis --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code Quality
lint:
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Running applications
run-streamlit:
	python scripts/run_streamlit.py

run-api:
	uvicorn sentiment_analysis.api.main:app --reload --host 0.0.0.0 --port 8000

# Docker
docker-build:
	docker build -t sentiment-analysis:latest -f docker/Dockerfile .

docker-run:
	docker run -p 8501:8501 -p 8000:8000 sentiment-analysis:latest

docker-build-dev:
	docker build -t sentiment-analysis:dev -f docker/Dockerfile.dev .

# Documentation
docs-serve:
	mkdocs serve

docs-build:
	mkdocs build

# Package
build:
	python -m build

# Pre-commit
pre-commit:
	pre-commit run --all-files
