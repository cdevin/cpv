# Plan Arithmetic: Compositional Plan Vectors for Multi-Task Control

[Project webpage](https://sites.google.com/berkeley.edu/compositionalplanvectors/home)

This codebase has been tested with python 3.5 and 3.6.
To install: clone the repository and run 
`pip install -r requirements.py`

Add the repo to you pythonpath by running
`export PYTHONPATH=[path/to/repo/]cpv/crafting:$PYTHONPATH'

To generate training data for the crafting environment, run

`cd [path/to/repo/]cpv/`

`mkdir data`

`python crafting/scripts/collect_composite_trajectories.py`

This may take a while.

Then, collect the evaluation trajectories by running

`python crafting/scripts/collect_reference_trajectories.py`

To train a CPV-Full model, run

`mkdir octresults`

`python crafting/gridworld/algorithms/cpv_experiments.py -H -P`

This script will print where the checkpoints are saved and where the tensorboard logs are saved.

To evaluate the model online, run

`python crafting/scripts/run_model_multitask_tensorboard.py --model [path/to/checkpoints_dir] --tb [path/to/tensorboard_dir] --type V3`


