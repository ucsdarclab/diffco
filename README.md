# DiffCo
This is the codebase for our paper **DiffCo: Auto-Differentiable Proxy Collision Detection with Multi-class Labels for Safety-Aware Trajectory Optimization**, Yuheng Zhi, Nikhil Das, Michael Yip. [[arxiv]]()

## Installation
It is recommended to use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) for virtual environment management.
```
pip install -r requirements.txt
pip install -e .
```

## Start using DiffCo:
This is a library of a relatively easy implementation of the differentiable collision checker itself (under the directory `diffco`) and some scripts to reproduce experiments in the paper (under the directory `script`)
1. `speed_compare.py` includes comprehensive examples of using DiffCo to do trajectory optimization. Besides the experiment in the paper, we also included an implementation of using DiffCo with Adam to optimize trajectories under constraints. 
2. `[2,3]d_data_generation.py` implements how to use MoveIt and FCL to generate the initial dataset for Baxter and Plannar robots, respectively. You may use any of your favourite collision detection library to do this.
3. `active.py` contains code of the active learning experiment.
4. `distest_error_vis.py` contains code for a few small experiments/demonstrations in the paper.
