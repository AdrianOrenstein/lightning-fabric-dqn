# Docker project template

```bash
# optionally build it yourself, but you should change config.yaml
make build fabric_dqn

# otherwise, docker pull adrianorenstein/fabric_dqn:latest, then,
make run fabric_dqn

# arg parsing with tyro
PYTHONPATH=$PWD:$PYTHONPATH python src/main.py --help

# running dqn
PYTHONPATH=$PWD:$PYTHONPATH python src/main.py dqn

# running the option dqn
PYTHONPATH=$PWD:$PYTHONPATH python src/main.py option_dqn
```