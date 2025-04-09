PYTHON = python3
PIP = pip3

.DEFAULT_GOAL = run

build:
	@if [ "$(filter apptainer,$(MAKECMDGOALS))" ]; then \
		./scripts/build_apptainer.sh $(filter-out $@,$(MAKECMDGOALS)); \
	else \
		./scripts/build_docker.sh $(filter-out $@,$(MAKECMDGOALS)); \
	fi

run:
	./scripts/run.sh $(filter-out $@, $(MAKECMDGOALS))

runblt: build lint test
	./scripts/run.sh pytorch $(filter-out $@, $(MAKECMDGOALS))

stop:
	@docker container kill $$(docker ps -q)

jupyter:
	./scripts/jupyter.sh

lint:
	./scripts/lint.sh

test:
	./scripts/test.sh

benchmark:
	./scripts/benchmark.sh

%:
	@:

