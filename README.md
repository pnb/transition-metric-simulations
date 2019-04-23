# transition-metric-simulations

## Python scripts for calculating transition probabilities and other measures of sequential association

This software is intended for research on sequential patterns of discrete states. For example, in behavioral research these states may be actions, emotions, or others. Measures of how likely one state is to follow another are often referred to as _transition metrics_. The functions in these scripts provide methods for:

1. Quantifying the propensity of one state to follow another (calculating transition metrics)
2. Simulating sequences with random states to study the properties of transition metrics and to aid in research study design (e.g., by deciding how much data to collect)

## Usage

The Python scripts in this repository can typically be downloaded and run without further installation. They require Python 3 and the `numpy` package. The simulation script (`simulate.py`) also requires the `matplotlib` package, and the `dot` executable (from [https://graphviz.org/](Graphviz)) must be in the system path or local directory.

`transition_metrics.py` has a command-line interface that can be used to calculate various transition metrics for your data. Run `python3 transition_metrics.py --help` to see usage information. This script can also be included in other Python programs if needed, as is the case for the simulation script.

Running simulations with random sequences via `simulate.py` requires modifying the script to change the length of sequences, number of simulations, number of states, base rates of states, etc. for your particular needs.
