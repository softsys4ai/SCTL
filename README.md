SCTL
==============================

A robust methodology to causal learning domain invariant feature set from a dataset without utilizing underlying graph structure. 
Paper for the method: https://arxiv.org/abs/2103.00139

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── Synthetic       <- Data that has been generated using R scripts.
    │   └── Real            <- Data that had been obtained from a real-world scenario.
    │
    ├── docs              <- A default folder containing the inital results and steps to run the code
    │
    ├── references         <- Python files containing implemetations of other methods
    ├── reports
    │   ├── figures        <- Generated normalized, selective plots used for reporting.
    │   ├── supp           <- All plots and values for each experiment for each setting.
    │   └── Python plots   <- Plots for all experimental settings generated in python.
    │                         
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    |
    ├── SCTL_experiments.ipynb   <- A sample notebook containing experimentation on multiple domain and comparison of SCTL with multiple methods 
    │    
    ├── src                <- Source code for use in this project.
    │   ├── _pycache_      <- Necessary files for python module
    │   │
    │   ├── data           <- R Scripts to generate data, t-test values and ci-tests.
    │   │   ├── data.R
    │   │   ├── data_dis_notequal.R
    │   │   ├── graph.R
    │   │   └── tests and final plots.R
    │   │
    │   ├── models       <- Scripts for raw versions Greedy subset search (GSS) and Exahustive Subset Search (NIPS) version we used
    │   │   ├── GSS_NIPS_model.py
    │   │   ├── ESS_model_(prediction script).py
    │   │   └── ESS_model_(original script).py
    │   │
    │   ├── all experiements+vizualization   <- Complete automated script generating all necessary experiments provided in the paper (values need to be fed for ESS)
    │   │   ├── outputs            <- supplmentary files and subfiles 
    │   │   ├── fast_cmim.py
    │   │   ├── utils.py
    │   │   ├── condense.py          <- MAIN FILE - generates plots and runs all experiments 
    │   │   ├── c45.py
    │   │   ├── FCBF_module.py
    │   │   ├── test.py
    │   │   ├── real_data_experiments.py
    │   │   └── real_data_visuals.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
