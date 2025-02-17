SHELL := /bin/bash

.PHONY: start-ui start-app start-docker stop-docker ui-deps app-deps all clean help

all: start-docker start-app

start-ui:
	cd ui && pnpm dev

start-app:
	poetry run uvicorn app.main:app --reload

start-docker:
	sudo docker compose up -d

stop-docker:
	sudo docker compose down

ui-deps:
	pnpm install

app-deps:
	poetry install --no-root

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:' Makefile | sed 's/://' | awk '{print "  ", $$0}'