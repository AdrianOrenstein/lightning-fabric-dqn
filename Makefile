PYTHON = python3
PIP = pip3

.DEFAULT_GOAL = run

build:
	./scripts/build_docker.sh $(filter-out $@,$(MAKECMDGOALS)); \

run:
	./scripts/run.sh $(filter-out $@, $(MAKECMDGOALS))

jupyter:
	./scripts/jupyter.sh

lint:
	./scripts/lint.sh

test:
	./scripts/test.sh
