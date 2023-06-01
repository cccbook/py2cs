set -x #echo on
python env_run.py CartPole-v1 human
python env_run.py FrozenLake-v1 human
python env_run.py FrozenLake-v1 rgb_array
