.ONESHELL:
SHELL := /bin/bash

RM = rm -fr

VENV_DIR := .venv
PYTHON := python3
PIP := pip
C_DIR := pwd

UV := $(shell command -v uv 2>/dev/null)

# ARGS := $(wordlist 2, 999, $(MAKECMDGOALS))

define ACTIVATE_VENV
if [ -z "$$VIRTUAL_ENV" ]; then \
	echo "No virtual environment detected."; \
	$(MAKE) venv; \
	. "$(VENV_DIR)/bin/activate"; \
fi
endef


help:
	uv run python -m src --help

install:
	set -e
	uv sync --locked
	npm ci --ignore-scripts

run:
	# uv run python -m src $(ACOMMAND) $(ARGS)
	# uv run python -m src

	uv run python -m agent_mbpp --task-file cache/mbpp_task.json \
		--output cache/mbpp_solution.json \
		--model-name "openrouter/free" \
		--provider-url "https://openrouter.ai/api/v1"
#		--provider-api-keys-file "cache/mbpp_task.json" 
# 		--provider-api-key "1 zsdfsd" \
# 		--provider-api-key "2 zsdfsd"
# 
#		--model-name "openrouter/free" "google/gemma-4-31b-it:free" \
#		--model-name "google/gemini-3.8-flash" not free\


debug:
	uv run python -m pdb -m src


check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "No virtual environment detected."; \
		$(MAKE) venv; \
		. "$(VENV_DIR)/bin/activate"; \
		echo "Virtual environment in $(VENV_DIR)."; \
	else \
		echo "Using virtual environment: $$VIRTUAL_ENV"; \
	fi

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		make install \
		chmod +x "$(VENV_DIR)/bin/activate" ; \
	else \
		echo "Virtual environment already exists."; \
	fi

clean:
	@$(RM) .mypy_cache
	@$(RM) __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name .pytest_cache -exec rm -rf {} +
	find . -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@$(RM) data/cache/

fclean: clean
	@$(RM) .cache
	@$(RM) .venv

lint: flake8 mypy

test:
	uv run --frozen python -m unittest discover -s tests -v

test-docker:
	set -e
	docker build -t agent-smith-swe-fixture -f tests/docker/Dockerfile tests/docker
	SWE_DOCKER_TEST_IMAGE=agent-smith-swe-fixture uv run --frozen python \
		-m unittest discover -s tests -p 'test_*docker*.py' -v

format:
	git ls-files -z -- '*.py' | xargs -0 uv --no-config tool run \
	--from autopep8==2.3.2 autopep8 --ignore-local-config \
	--global-config /dev/null --max-line-length 79 --in-place

flake8:
	git ls-files -z -- '*.py' | xargs -0 uv run --frozen flake8

mypy:
	git ls-files -z -- '*.py' | xargs -0 uv run --frozen mypy --warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs \
	--explicit-package-bases \
	--follow-imports=silent

lint-strict: flake8
	git ls-files -z -- '*.py' | xargs -0 uv run --frozen mypy \
	--strict --explicit-package-bases --follow-imports=silent


.PHONY: install run debug clean format lint lint-strict flake8 mypy test test-docker fclean runapi check-venv venv

#  ln -s ~/goinfre/uv uv
