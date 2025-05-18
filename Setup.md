### Setup environment
#### Conda python setup

```shell
conda create -n mlagents python=3.10.12
conda activate mlagents
```

#### Local Installation for Development
From the repository's `Install` directory, run:
```shell
pip3 install torch -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
```
Running pip with the -e flag will let you make changes to the Python files directly and have those reflected when you run mlagents-learn. It is important to install these packages in this order as the mlagents package depends on mlagents_envs, and installing it in the other order will download mlagents_envs from PyPi.

### Train environment

```shell
mlagents-learn ./Configuration/3v3_Soccer.yaml --env <env-path> --num-envs <n> --run-id <run-identifier> --no-graphics --torch-device cuda
```