# Makefile for Hospital Readmission Dashboard

.PHONY: help install run test clean docker-build docker-run

help:
	@echo "Hospital Readmission Dashboard - Available Commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make run          - Run Streamlit dashboard"
	@echo "  make api          - Run Flask API"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean generated files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run with Docker Compose"
	@echo "  make train        - Train ML models"
	@echo "  make download     - Download dataset"
	@echo ""

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

api:
	python api.py

test:
	python -m pytest tests/ -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete

docker-build:
	docker build -t hospital-dashboard .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

train:
	python src/train_models.py

download:
	python data/download_data.py

setup:
	chmod +x setup.sh
	./setup.sh
