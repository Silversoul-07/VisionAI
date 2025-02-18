SHELL := /bin/bash

.PHONY: start-ui start-app start-docker stop-docker ui-deps app-deps all clean help docker-clean poetry-clean

all: start-docker start-app

clean: docker-clean poetry-clean
	@echo "🧹 Cleaning all environments and dependencies..."

start-ui:
	cd ui && pnpm dev

start-app:
	poetry install --no-root && poetry run uvicorn app.main:app --reload

start-docker:
	sudo docker compose up -d

stop-docker:
	sudo docker compose down

ui-deps:
	pnpm install

app-deps:
	poetry install --no-root

docker-clean:
	@echo "🐳 Cleaning Docker resources..."
	sudo docker compose down --rmi all --volumes --remove-orphans
	sudo docker system prune -af

poetry-clean:
	@echo "📦 Cleaning Poetry environment and dependencies..."
	poetry env remove --all

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:' Makefile | sed 's/://' | awk '{print "  ", $$0}'